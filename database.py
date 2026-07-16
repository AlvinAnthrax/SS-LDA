import psycopg2
import psycopg2.extras
import psycopg2.extensions
import pandas as pd
from contextlib import contextmanager
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration from .env
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'db_Review'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'port': os.getenv('DB_PORT', '5432')
}


@contextmanager
def get_connection():
    """Context manager untuk database connection dengan auto-close dan error handling."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("PostgreSQL connection established successfully")
        yield conn
    except psycopg2.OperationalError as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_connection: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("Connection closed")


@contextmanager
def get_cursor(conn):
    """Context manager untuk cursor dengan auto-close."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield cur
    finally:
        cur.close()


def get_safe_table_name(package_id):
    """Sanitize table name (replace dots with underscores)."""
    if not package_id or not isinstance(package_id, str):
        raise ValueError("package_id must be a non-empty string")
    safe_name = package_id.replace('.', '_').replace('-', '_').lower()
    if not safe_name.replace('_', '').isalnum():
        raise ValueError(f"Invalid package_id: {package_id}")
    return safe_name


def validate_review_data(username, content, score):
    """Validate review data before insertion."""
    if not isinstance(username, str) or not username.strip():
        raise ValueError("Username must be a non-empty string")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Content must be a non-empty string")
    if not isinstance(score, int) or score < 0 or score > 5:
        raise ValueError("Score must be an integer between 0-5")
    return True


def init_db_table(package_id):
    """Create table if not exists."""
    safe_name = get_safe_table_name(package_id)
    
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {safe_name} (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                content TEXT NOT NULL,
                score INTEGER NOT NULL CHECK (score >= 0 AND score <= 5),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(username, content)
            );
            """
            try:
                cur.execute(create_table_sql)
                conn.commit()
                logger.info(f"Table '{safe_name}' created/verified successfully")

                # Create indexes separately to avoid issues if columns are missing
                try:
                    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{safe_name}_username ON {safe_name}(username);")
                except psycopg2.Error:
                    logger.warning(f"Could not create index idx_{safe_name}_username yet")
                try:
                    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{safe_name}_score ON {safe_name}(score);")
                except psycopg2.Error:
                    logger.warning(f"Could not create index idx_{safe_name}_score yet")
                try:
                    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{safe_name}_created_at ON {safe_name}(created_at);")
                except psycopg2.Error:
                    logger.warning(f"Could not create index idx_{safe_name}_created_at yet")
                conn.commit()
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Error creating table '{safe_name}': {e}")
                raise


def save_reviews(package_id, df):
    """Batch insert reviews (optimized with executemany)."""
    if df is None or df.empty:
        logger.warning("Empty dataframe provided to save_reviews")
        return 0
    
    safe_name = get_safe_table_name(package_id)
    init_db_table(package_id)
    
    # Prepare data
    data_to_insert = []
    validation_errors = []
    
    for idx, row in df.iterrows():
        try:
            username = str(row.get('userName', '')).strip()
            content = str(row.get('content', '')).strip()
            score = int(row.get('score', 0))
            
            validate_review_data(username, content, score)
            data_to_insert.append((username, content, score))
        except (ValueError, TypeError) as e:
            validation_errors.append(f"Row {idx}: {str(e)}")
            continue
    
    if validation_errors:
        logger.warning(f"Validation errors found: {validation_errors}")
    
    if not data_to_insert:
        logger.warning("No valid data to insert after validation")
        return 0
    
    # Batch insert
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            query = f"""
            INSERT INTO {safe_name} (username, content, score)
            VALUES (%s, %s, %s)
            ON CONFLICT (username, content) DO NOTHING;
            """
            try:
                psycopg2.extras.execute_batch(cur, query, data_to_insert, page_size=1000)
                conn.commit()
                inserted = len(data_to_insert)
                logger.info(f"Successfully inserted {inserted} reviews into '{safe_name}'")
                return inserted
            except psycopg2.IntegrityError as e:
                conn.rollback()
                logger.error(f"Integrity error while inserting: {e}")
                return 0
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Database error while inserting: {e}")
                raise


def load_reviews(package_id, limit=None):
    """Load reviews from table."""
    safe_name = get_safe_table_name(package_id)
    
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            query = f"""
            SELECT id, username, content, score, created_at 
            FROM {safe_name} 
            WHERE content IS NOT NULL AND content != ''
            ORDER BY created_at DESC
            """
            if limit:
                query += f" LIMIT {int(limit)}"

            try:
                cur.execute(query)
                rows = cur.fetchall()
                logger.info(f"Loaded {len(rows)} reviews from '{safe_name}'")
                return [row['content'] for row in rows]
            except psycopg2.Error as e:
                # Handle missing columns (created_at/updated_at) by attempting to add them
                err_msg = str(e).lower()
                if 'created_at' in err_msg or 'updated_at' in err_msg:
                    logger.warning(f"Missing timestamp column in '{safe_name}', attempting to add columns: {e}")
                    try:
                        # clear current failed transaction state
                        conn.rollback()
                    except Exception:
                        pass
                    try:
                        # Add columns safely if they do not exist
                        with get_cursor(conn) as cur2:
                            cur2.execute(f"ALTER TABLE {safe_name} ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
                            cur2.execute(f"ALTER TABLE {safe_name} ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
                        conn.commit()
                        logger.info(f"Added missing timestamp columns to '{safe_name}', retrying load_reviews")
                        with get_cursor(conn) as cur3:
                            cur3.execute(query)
                            rows = cur3.fetchall()
                        return [row['content'] for row in rows]
                    except psycopg2.Error as e2:
                        conn.rollback()
                        logger.error(f"Failed to add missing columns to '{safe_name}': {e2}")
                        raise
                else:
                    logger.error(f"Error loading reviews from '{safe_name}': {e}")
                    raise


def load_reviews_detailed(package_id, limit=None):
    """Load reviews with all details (id, username, score, timestamp)."""
    safe_name = get_safe_table_name(package_id)
    
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            query = f"""
            SELECT id, username, content, score, created_at, updated_at
            FROM {safe_name}
            WHERE content IS NOT NULL AND content != ''
            ORDER BY created_at DESC
            """
            if limit:
                query += f" LIMIT {int(limit)}"
            
            try:
                cur.execute(query)
                rows = cur.fetchall()
                logger.info(f"Loaded {len(rows)} detailed reviews from '{safe_name}'")
                return [dict(row) for row in rows]
            except psycopg2.Error as e:
                err_msg = str(e).lower()
                if 'created_at' in err_msg or 'updated_at' in err_msg:
                    logger.warning(f"Missing timestamp column in '{safe_name}', attempting to add columns: {e}")
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    try:
                        with get_cursor(conn) as cur2:
                            cur2.execute(f"ALTER TABLE {safe_name} ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
                            cur2.execute(f"ALTER TABLE {safe_name} ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
                        conn.commit()
                        logger.info(f"Added missing timestamp columns to '{safe_name}', retrying load_reviews_detailed")
                        with get_cursor(conn) as cur3:
                            cur3.execute(query)
                            rows = cur3.fetchall()
                        return [dict(row) for row in rows]
                    except psycopg2.Error as e2:
                        conn.rollback()
                        logger.error(f"Failed to add missing columns to '{safe_name}': {e2}")
                        raise
                else:
                    logger.error(f"Error loading detailed reviews: {e}")
                    raise


def update_review(package_id, review_id, content=None, score=None):
    """Update existing review (UPDATE operation)."""
    safe_name = get_safe_table_name(package_id)
    
    if content is None and score is None:
        logger.warning("No fields to update")
        return False
    
    update_fields = []
    update_values = []
    
    if content is not None:
        content = str(content).strip()
        if not content:
            raise ValueError("Content cannot be empty")
        update_fields.append("content = %s")
        update_values.append(content)
    
    if score is not None:
        score = int(score)
        if score < 0 or score > 5:
            raise ValueError("Score must be between 0-5")
        update_fields.append("score = %s")
        update_values.append(score)
    
    update_values.append(review_id)
    
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            query = f"""
            UPDATE {safe_name}
            SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            RETURNING id, username, content, score, updated_at;
            """
            try:
                cur.execute(query, update_values)
                result = cur.fetchone()
                conn.commit()
                
                if result:
                    logger.info(f"Review {review_id} updated successfully")
                    return True
                else:
                    logger.warning(f"Review {review_id} not found")
                    return False
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Error updating review: {e}")
                raise


def delete_review(package_id, review_id):
    """Delete review by ID (DELETE operation)."""
    safe_name = get_safe_table_name(package_id)
    
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            query = f"""
            DELETE FROM {safe_name}
            WHERE id = %s
            RETURNING id;
            """
            try:
                cur.execute(query, (review_id,))
                result = cur.fetchone()
                conn.commit()
                
                if result:
                    logger.info(f"Review {review_id} deleted successfully")
                    return True
                else:
                    logger.warning(f"Review {review_id} not found")
                    return False
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Error deleting review: {e}")
                raise


def delete_reviews_by_criteria(package_id, score=None, username=None):
    """Delete multiple reviews by criteria."""
    safe_name = get_safe_table_name(package_id)
    
    if score is None and username is None:
        raise ValueError("Must specify at least one criteria (score or username)")
    
    conditions = []
    params = []
    
    if score is not None:
        conditions.append("score = %s")
        params.append(score)
    
    if username is not None:
        conditions.append("username = %s")
        params.append(username)
    
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            query = f"""
            DELETE FROM {safe_name}
            WHERE {' AND '.join(conditions)}
            RETURNING id;
            """
            try:
                cur.execute(query, params)
                deleted_ids = cur.fetchall()
                conn.commit()
                deleted_count = len(deleted_ids)
                logger.info(f"Deleted {deleted_count} reviews from '{safe_name}'")
                return deleted_count
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Error deleting reviews: {e}")
                raise


def get_review_statistics(package_id):
    """Get statistics about reviews (count, average score, etc)."""
    safe_name = get_safe_table_name(package_id)
    
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            query = f"""
            SELECT 
                COUNT(*) as total_reviews,
                COUNT(DISTINCT username) as unique_users,
                AVG(CAST(score AS NUMERIC)) as avg_score,
                MIN(score) as min_score,
                MAX(score) as max_score,
                MIN(created_at) as first_review,
                MAX(created_at) as last_review
            FROM {safe_name};
            """
            try:
                cur.execute(query)
                stats = cur.fetchone()
                logger.info(f"Retrieved statistics for '{safe_name}'")
                return dict(stats)
            except psycopg2.Error as e:
                logger.error(f"Error getting statistics: {e}")
                raise


def init_db(package_id=None):
    """Initialize database table."""
    if package_id:
        init_db_table(package_id)
    else:
        logger.warning("No package_id provided for initialization")


if __name__ == '__main__':
    logger.info("Database module ready for PostgreSQL with full CRUD operations")
