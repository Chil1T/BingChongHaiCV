import React from 'react';
import Uploader from './components/Uploader';
import { Layout, Typography } from 'antd';

const { Header, Content } = Layout;
const { Title } = Typography;

const App: React.FC = () => {
  const handleResult = (result: any) => {
    console.log('识别结果:', result);
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', padding: '0 50px' }}>
        <Title level={2} style={{ margin: '16px 0' }}>植物病虫害识别系统</Title>
      </Header>
      <Content style={{ padding: '0 50px', marginTop: 32 }}>
        <div style={{ background: '#fff', padding: 24, minHeight: 280 }}>
          <Uploader testId="uploader" className="main-uploader" onResult={handleResult} />
        </div>
      </Content>
    </Layout>
  );
};

export default App;