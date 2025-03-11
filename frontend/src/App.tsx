import React, { useState } from 'react';
import Uploader from './components/Uploader';
import ResultDisplay from './components/ResultDisplay';
import { Layout, Typography, Card, Row, Col, Statistic } from 'antd';
import { ExperimentOutlined, RocketOutlined } from '@ant-design/icons';

const { Header, Content, Footer } = Layout;
const { Title } = Typography;

interface Prediction {
  class: string;
  probability: number;
}

interface ResultData {
  predictions: Prediction[];
}

interface Result {
  status: number;
  data: ResultData;
  inference_time: number;
  error?: string;
}

const App: React.FC = () => {
  const [result, setResult] = useState<Result | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [totalPredictions, setTotalPredictions] = useState<number>(0);

  const handleResult = (resultData: Result, imageUrl: string) => {
    setResult(resultData);
    setImageUrl(imageUrl);
    setTotalPredictions(prev => prev + 1);
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', padding: '0 50px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Title level={2} style={{ margin: '16px 0' }}>植物病虫害识别系统</Title>
          <div style={{ display: 'flex', gap: '20px' }}>
            <Statistic 
              title="识别次数" 
              value={totalPredictions} 
              prefix={<ExperimentOutlined />} 
              valueStyle={{ color: '#3f8600' }}
            />
            <Statistic 
              title="支持疾病类型" 
              value={38} 
              prefix={<RocketOutlined />} 
              valueStyle={{ color: '#1890ff' }}
            />
          </div>
        </div>
      </Header>
      <Content style={{ padding: '0 50px', marginTop: 32 }}>
        <Row gutter={[24, 24]}>
          <Col xs={24} md={12}>
            <Card title="上传图片" bordered={false}>
              <Uploader testId="uploader" className="main-uploader" onResult={handleResult} />
              
              {imageUrl && (
                <div style={{ marginTop: 20, textAlign: 'center' }}>
                  <img
                    src={imageUrl}
                    alt="上传的图片"
                    style={{ maxWidth: '100%', maxHeight: 300 }}
                  />
                </div>
              )}
            </Card>
          </Col>
          
          <Col xs={24} md={12}>
            <ResultDisplay result={result} />
          </Col>
        </Row>
      </Content>
      <Footer style={{ textAlign: 'center' }}>
        植物病虫害识别系统 ©{new Date().getFullYear()} 基于深度学习的智能诊断
      </Footer>
    </Layout>
  );
};

export default App;