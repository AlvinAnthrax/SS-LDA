import { Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "antd";
import Dashboard from "./pages/Dashboard";
import "./App.css";

const { Header, Content } = Layout;

function App() {
  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Header className="app-header">
        <div className="logo">SS-LDA Dashboard</div>
      </Header>
      <Content className="app-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </Content>
    </Layout>
  );
}

export default App;
