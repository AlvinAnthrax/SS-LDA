from google_play_scraper import app
from google_play_scraper import Sort, reviews
import pandas as pd

from shortener import extract_package_id
from database import save_reviews

def scrap(token):
    package_id = extract_package_id(token)

    print(token)
    print(f"Extracted package_id: {package_id}")
    if not package_id:
        print("Invalid package_id extracted from token")
        return pd.DataFrame()
    
    all_reviews = []

    # Scrape reviews for each score (1-5)
    for i in range(1, 6):
        print(f"Scraping score {i}...")
        result, continuation_token = reviews(
            package_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=5000,
            filter_score_with=i
        )
        all_reviews.extend(result)
        print(f"Got {len(result)} reviews for score {i}")
    
    print('Scraping complete')
    df = pd.DataFrame(all_reviews)

    if len(df) > 0:
        print(f"Saving {len(df)} reviews for {package_id}")
        save_reviews(package_id, df)
        return df
    else:
        print("No reviews found or wrong ID/link, please check again")
        return pd.DataFrame()

# Example usage
# if __name__ == "__main__":
#     # token = "https://play.google.com/store/apps/details?id=com.mobile.legends&hl=id"
#     token = "https://play.google.com/store/apps/details?id=com.whatsapp&hl=id"
#     result = scrap(token)
#     print(result.head())
