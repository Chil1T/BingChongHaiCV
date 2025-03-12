import React, { useState } from 'react';
import { Layout, Space } from 'antd';
import Uploader from './components/Uploader';
import ResultDisplay from './components/ResultDisplay';
import DiseaseGuide from './components/DiseaseGuide';
import HistoryRecord from './components/HistoryRecord';
import { PredictionResponse } from './api/client';
import './App.css';

const { Header, Content } = Layout;

interface HistoryItem {
  id: string;
  timestamp: string;
  imageUrl: string;
  result: PredictionResponse;
}

const App: React.FC = () => {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  const handleResult = (newResult: PredictionResponse, imageUrl: string) => {
    setResult(newResult);
    
    // 添加到历史记录
    const historyItem: HistoryItem = {
      id: Date.now().toString(),
      timestamp: new Date().toLocaleString(),
      imageUrl,
      result: newResult
    };
    setHistory(prev => [historyItem, ...prev]);
  };

  return (
    <Layout className="app-container">
      <Header className="app-header">
        <h1>植物病虫害识别系统</h1>
        <Space size="middle">
          <DiseaseGuide />
          <HistoryRecord records={history} />
        </Space>
      </Header>
      <Content className="app-content">
        <div className="main-container">
          <Uploader
            testId="plant-disease-uploader"
            onResult={handleResult}
          />
          <ResultDisplay result={result} />
        </div>
      </Content>
    </Layout>
  );
};

export default App;