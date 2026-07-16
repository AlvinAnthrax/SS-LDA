import { useMemo, useState } from "react";
import {
  Card,
  Row,
  Col,
  Button,
  Input,
  Typography,
  Space,
  Alert,
  Table,
  Statistic,
  Tag,
  List,
} from "antd";
import axios from "axios";

const { Title, Text } = Typography;
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const defaultCounts = {
  raw_reviews: 0,
  neutral: 0,
  positive: 0,
  negative: 0,
};

const parseTopicDistribution = (value) => {
  if (!value || typeof value !== "string") return [];

  return value
    .split("+")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => {
      const match = item.match(/([0-9.]+)\*"(.*?)"/);
      if (!match) return null;
      return {
        weight: Number(match[1]),
        word: match[2],
      };
    })
    .filter(Boolean);
};

function Dashboard() {
  const [packageId, setPackageId] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);

  const handleAnalyze = async () => {
    if (!packageId.trim()) {
      setError("Silakan masukkan package_id atau URL aplikasi");
      return;
    }

    setError("");
    setLoading(true);
    setResult(null);
    setProgress(10);

    try {
      const raw = (packageId || "").trim();
      const match = raw.match(/[?&]id=([^&\s]+)/);
      const pkg = match ? decodeURIComponent(match[1]) : raw;

      setProgress(40);
      const response = await axios.post(`${API_BASE_URL}/analyze`, {
        package_id: pkg,
        sample_size: 100,
        use_multiprocessing: true,
      });
      setProgress(100);
      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail || "Terjadi kesalahan saat memproses data",
      );
    } finally {
      setLoading(false);
      setTimeout(() => setProgress(0), 500);
    }
  };

  const columns = [
    { title: "Topik", dataIndex: "topic", key: "topic" },
    { title: "Kata Utama", dataIndex: "words", key: "words" },
  ];

  const buildTopicRows = (model) => {
    if (!model || !model.topic_word_dist) return [];
    return Object.entries(model.topic_word_dist).map(([topic, dist]) => ({
      key: topic,
      topic: `Topik ${topic}`,
      words:
        parseTopicDistribution(dist)
          .slice(0, 5)
          .map((item) => `${item.word} (${item.weight.toFixed(2)})`)
          .join(", ") || dist,
    }));
  };

  const sentimentSummary = useMemo(() => {
    if (!result?.data_counts) return defaultCounts;
    return result.data_counts;
  }, [result]);

  const dominantSentiment = useMemo(() => {
    const counts = [
      { key: "positive", value: sentimentSummary.positive },
      { key: "neutral", value: sentimentSummary.neutral },
      { key: "negative", value: sentimentSummary.negative },
    ];
    const best = counts.reduce((current, candidate) =>
      candidate.value > current.value ? candidate : current,
    );
    return best.key;
  }, [sentimentSummary]);

  const sentimentInsights = useMemo(() => {
    if (!result?.models) return [];

    return Object.entries(result.models).map(([label, model]) => {
      const topicEntries = Object.entries(model?.topic_word_dist || {}).map(
        ([topic, dist]) => ({
          topic,
          words: parseTopicDistribution(dist).slice(0, 3),
        }),
      );

      const dominantTopic = topicEntries.reduce(
        (best, current) => {
          const bestWeight = best.words[0]?.weight || 0;
          const currentWeight = current.words[0]?.weight || 0;
          return currentWeight > bestWeight ? current : best;
        },
        topicEntries[0] || { topic: "-", words: [] },
      );

      const coherenceValues = model?.coherence_score
        ? Object.values(model.coherence_score)
        : [];
      const averageCoherence =
        coherenceValues.length > 0
          ? (
              coherenceValues.reduce(
                (sum, value) => sum + Number(value || 0),
                0,
              ) / coherenceValues.length
            ).toFixed(3)
          : "-";

      return {
        key: label,
        label,
        headline: `${label.charAt(0).toUpperCase() + label.slice(1)} menonjol pada topik ${dominantTopic.topic}`,
        detail: `Kata kunci dominan: ${
          dominantTopic.words.map((item) => item.word).join(", ") || "-"
        }`,
        topWords: dominantTopic.words.map((item) => item.word),
        averageCoherence,
      };
    });
  }, [result]);

  const sentimentBarData = useMemo(() => {
    return [
      { label: "Neutral", value: sentimentSummary.neutral, color: "#1677ff" },
      { label: "Positive", value: sentimentSummary.positive, color: "#52c41a" },
      { label: "Negative", value: sentimentSummary.negative, color: "#f5222d" },
    ];
  }, [sentimentSummary]);

  const sentimentPieData = useMemo(() => {
    const total =
      sentimentSummary.raw_reviews > 0 ? sentimentSummary.raw_reviews : 1;
    let cursor = 0;

    return sentimentBarData.map((item) => {
      const start = cursor;
      const end = cursor + (item.value / total) * 100;
      cursor = end;
      return { ...item, start, end };
    });
  }, [sentimentBarData, sentimentSummary.raw_reviews]);

  return (
    <div className="dashboard-container">
      <Card className="dashboard-card" bordered={false}>
        <Title level={3}>Analisis Review Google Play</Title>
        <Space direction="vertical" size="middle" style={{ width: "100%" }}>
          <Input
            placeholder="Masukkan package_id atau URL Google Play"
            value={packageId}
            onChange={(e) => setPackageId(e.target.value)}
            onPressEnter={handleAnalyze}
          />
          {loading && (
            <div className="progress-block">
              <Text type="secondary">Memproses analisis, harap tunggu...</Text>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
          <Button
            type="primary"
            loading={loading}
            onClick={handleAnalyze}
            block
          >
            Jalankan Analisis
          </Button>
          {error && <Alert message={error} type="error" showIcon />}
        </Space>
      </Card>

      {result && (
        <div className="result-section">
          <Row gutter={[16, 16]}>
            <Col span={8}>
              <Card title="Rekap Sentimen" bordered={false}>
                <Space
                  direction="vertical"
                  size="small"
                  style={{ width: "100%" }}
                >
                  <Statistic
                    title="Sentimen dominan"
                    value={dominantSentiment}
                  />
                  <Text type="secondary">
                    Neutral: {sentimentSummary.neutral} | Positive:{" "}
                    {sentimentSummary.positive} | Negative:{" "}
                    {sentimentSummary.negative}
                  </Text>
                </Space>
              </Card>
            </Col>
            <Col span={8}>
              <Card title="Ringkasan SS-LDA" bordered={false}>
                <Space
                  direction="vertical"
                  size="small"
                  style={{ width: "100%" }}
                >
                  <Statistic
                    title="Total review diproses"
                    value={sentimentSummary.raw_reviews}
                  />
                  <Text type="secondary">
                    Model membagi data menjadi topik yang bisa dipelajari per
                    sentimen.
                  </Text>
                </Space>
              </Card>
            </Col>
            <Col span={8}>
              <Card title="Insight Utama" bordered={false}>
                <Space
                  direction="vertical"
                  size="small"
                  style={{ width: "100%" }}
                >
                  {sentimentInsights.map((item) => (
                    <div key={item.key}>
                      <Text strong>{item.label}</Text>
                      <br />
                      <Text type="secondary">{item.detail}</Text>
                    </div>
                  ))}
                </Space>
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
            <Col span={24}>
              <Card title="Distribusi Sentimen" bordered={false}>
                <div className="pie-chart-card">
                  <div className="pie-chart-legend">
                    {sentimentBarData.map((item) => (
                      <div key={item.label} className="legend-item">
                        <span
                          className="legend-dot"
                          style={{ backgroundColor: item.color }}
                        />
                        <span>{item.label}</span>
                      </div>
                    ))}
                  </div>
                  <div className="pie-chart">
                    <div
                      className="pie-segment"
                      style={{
                        background:
                          sentimentSummary.raw_reviews > 0
                            ? `conic-gradient(${sentimentPieData
                                .map(
                                  (item) =>
                                    `${item.color} ${item.start}% ${item.end}%`,
                                )
                                .join(
                                  ", ",
                                )}, #f0f2f5 ${sentimentPieData[sentimentPieData.length - 1]?.end || 0}% 100%)`
                            : "#f0f2f5",
                      }}
                    >
                      <div className="pie-center">
                        <strong>{sentimentSummary.raw_reviews}</strong>
                        <span>review</span>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
            <Col span={24}>
              <Card title="Insight per Sentimen" bordered={false}>
                <List
                  itemLayout="horizontal"
                  dataSource={sentimentInsights}
                  renderItem={(item) => (
                    <List.Item>
                      <Space
                        direction="vertical"
                        size="small"
                        style={{ width: "100%" }}
                      >
                        <Space>
                          <Tag color="blue">{item.label}</Tag>
                          <Text strong>{item.headline}</Text>
                        </Space>
                        <Text type="secondary">
                          Koherensi rata-rata: {item.averageCoherence}
                        </Text>
                        <Space wrap>
                          {item.topWords.map((word) => (
                            <Tag key={word}>{word}</Tag>
                          ))}
                        </Space>
                      </Space>
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
            <Col span={12}>
              <Card title="Model Neutral" bordered={false}>
                <Table
                  columns={columns}
                  dataSource={buildTopicRows(result.models.neutral)}
                  pagination={false}
                  size="small"
                />
              </Card>
            </Col>
            <Col span={12}>
              <Card title="Model Positive" bordered={false}>
                <Table
                  columns={columns}
                  dataSource={buildTopicRows(result.models.positive)}
                  pagination={false}
                  size="small"
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
            <Col span={12}>
              <Card title="Model Negative" bordered={false}>
                <Table
                  columns={columns}
                  dataSource={buildTopicRows(result.models.negative)}
                  pagination={false}
                  size="small"
                />
              </Card>
            </Col>
            <Col span={12}>
              <Card title="Koherensi per Topik" bordered={false}>
                {Object.entries(
                  result.models.neutral.coherence_score || {},
                ).map(([key, value]) => (
                  <div key={key} style={{ marginBottom: 8 }}>
                    <Text strong>{key}:</Text> <Text>{value}</Text>
                  </div>
                ))}
              </Card>
            </Col>
          </Row>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
