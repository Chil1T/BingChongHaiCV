import React, { useState, useEffect, useCallback } from 'react';
import { Upload, message, Button, Spin, Alert, Card, Row, Col, Typography } from 'antd';
import { UploadOutlined, LoadingOutlined, EyeOutlined } from '@ant-design/icons';
import { uploadImage, checkHealth, HealthCheckResponse } from '../api/client';
import './Uploader.module.css';

const { Title } = Typography;

interface UploaderProps {
  className?: string;
  testId: string;
  onResult: (result: any, imageUrl: string) => void;
}

const Uploader: React.FC<UploaderProps> = ({ className = 'plant-disease-uploader', testId, onResult }) => {
  const [loading, setLoading] = useState(false);
  const [serviceStatus, setServiceStatus] = useState<HealthCheckResponse | null>(null);
  const [checkingStatus, setCheckingStatus] = useState(true);
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  // 清理预览图片URL
  const cleanupPreviewImage = useCallback(() => {
    if (previewImage) {
      URL.revokeObjectURL(previewImage);
    }
  }, [previewImage]);

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      cleanupPreviewImage();
    };
  }, [cleanupPreviewImage]);

  useEffect(() => {
    const checkServiceStatus = async () => {
      setCheckingStatus(true);
      try {
        const status = await checkHealth();
        setServiceStatus(status);
      } catch (error) {
        console.error('Service status check failed:', error);
      } finally {
        setCheckingStatus(false);
      }
    };

    checkServiceStatus();
    const interval = setInterval(checkServiceStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const refreshServiceStatus = async () => {
    setCheckingStatus(true);
    try {
      const status = await checkHealth();
      setServiceStatus(status);
    } catch (error) {
      console.error('Service status check failed:', error);
    } finally {
      setCheckingStatus(false);
    }
  };

  const beforeUpload = (file: File) => {
    const isImage = file.type.startsWith('image/');
    if (!isImage) {
      message.error('只能上传图片文件！');
      return false;
    }
    const isLt5M = file.size / 1024 / 1024 < 5;
    if (!isLt5M) {
      message.error('图片大小不能超过5MB！');
      return false;
    }

    // 清理旧的预览URL
    cleanupPreviewImage();
    
    // 创建新的预览URL
    const imageUrl = URL.createObjectURL(file);
    setPreviewImage(imageUrl);
    
    return true;
  };

  const handleUpload = async (file: File) => {
    if (!serviceStatus?.model_loaded) {
      message.error('服务正在初始化，请稍后再试！');
      return;
    }

    try {
      setLoading(true);
      const result = await uploadImage(file);
      
      // 清理旧的预览URL
      cleanupPreviewImage();
      
      // 创建新的预览URL
      const imageUrl = URL.createObjectURL(file);
      setPreviewImage(imageUrl);
      
      // 传递结果和图片URL
      onResult(result, imageUrl);
      message.success('识别成功！');
    } catch (error) {
      message.error('识别失败，请重试！');
      console.error('Upload error:', error);
    } finally {
      setLoading(false);
    }
  };

  if (checkingStatus) {
    return (
      <Card className={className} data-testid={testId}>
        <Spin indicator={<LoadingOutlined style={{ fontSize: 24 }} spin />}>
          <div style={{ padding: '20px', textAlign: 'center' }}>
            正在检查服务状态...
          </div>
        </Spin>
      </Card>
    );
  }

  if (!serviceStatus?.model_loaded) {
    return (
      <Card className={className} data-testid={testId}>
        <Alert
          message={serviceStatus?.loading_in_progress ? "模型加载中" : "服务正在初始化"}
          description={
            <div>
              <p>
                {serviceStatus?.loading_in_progress 
                  ? "模型正在后台加载中，这可能需要一些时间，请稍后再试..." 
                  : "模型正在初始化，请稍后再试..."}
              </p>
              {serviceStatus && (
                <div style={{ marginTop: 10 }}>
                  <p><strong>状态详情：</strong></p>
                  <ul>
                    <li>模式: {serviceStatus.mode === 'demo' ? '演示模式' : '生产模式'}</li>
                    {serviceStatus.model_info && (
                      <>
                        <li>模型路径: {serviceStatus.model_info.path}</li>
                        <li>模型文件存在: {serviceStatus.model_info.exists ? '是' : '否'}</li>
                        <li>模型大小: {serviceStatus.model_info.size_mb ? `${serviceStatus.model_info.size_mb} MB` : '未知'}</li>
                        <li>运行设备: {serviceStatus.model_info.device}</li>
                        {serviceStatus.loading_in_progress && (
                          <li><strong>状态: 正在加载中...</strong></li>
                        )}
                      </>
                    )}
                    {serviceStatus.numpy_info && (
                      <>
                        <li>NumPy版本: {serviceStatus.numpy_info.version || '未安装'}</li>
                        <li>NumPy兼容性: {serviceStatus.numpy_info.compatible ? '正常' : 
                          <span style={{ color: 'red' }}>
                            不兼容 - 请运行 fix_numpy.bat 修复
                          </span>
                        }</li>
                        {serviceStatus.numpy_info.error && (
                          <li>NumPy错误: {serviceStatus.numpy_info.error}</li>
                        )}
                      </>
                    )}
                    {serviceStatus.demo_mode_reason && (
                      <li>演示模式原因: {serviceStatus.demo_mode_reason}</li>
                    )}
                  </ul>
                  <Button 
                    type="primary" 
                    onClick={refreshServiceStatus}
                    style={{ marginTop: 10 }}
                  >
                    刷新状态
                  </Button>
                  {serviceStatus.loading_in_progress && (
                    <>
                      <p style={{ marginTop: 10, fontStyle: 'italic' }}>
                        提示: 模型较大(360MB+)，首次加载可能需要1-2分钟，请耐心等待
                      </p>
                      <Button 
                        type="default" 
                        onClick={() => {
                          setServiceStatus(prev => prev ? {
                            ...prev,
                            mode: 'demo',
                            model_loaded: true,
                            demo_mode_reason: '用户选择使用演示模式'
                          } : null);
                        }}
                        style={{ marginTop: 10, marginLeft: 10 }}
                        danger
                      >
                        使用演示模式
                      </Button>
                      <p style={{ marginTop: 5, fontSize: '12px', color: '#999' }}>
                        演示模式将使用随机生成的结果，仅用于演示界面功能
                      </p>
                    </>
                  )}
                </div>
              )}
            </div>
          }
          type="info"
          showIcon
        />
      </Card>
    );
  }

  return (
    <Card className={className} data-testid={testId}>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Title level={4} style={{ textAlign: 'center', marginBottom: 24 }}>
            植物病害智能识别系统
          </Title>
        </Col>
        
        <Col span={24}>
          <div style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center',
            gap: '16px'
          }}>
            {previewImage && (
              <div style={{
                width: '100%',
                maxWidth: '400px',
                marginBottom: '16px',
                position: 'relative'
              }}>
                <img
                  src={previewImage}
                  alt="预览图"
                  style={{
                    width: '100%',
                    height: 'auto',
                    borderRadius: '8px',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                  }}
                />
                {loading && (
                  <div style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    background: 'rgba(255, 255, 255, 0.7)',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    borderRadius: '8px'
                  }}>
                    <Spin size="large" />
                  </div>
                )}
              </div>
            )}
            
            <Upload
              accept="image/*"
              showUploadList={false}
              beforeUpload={beforeUpload}
              customRequest={({ file }) => handleUpload(file as File)}
              disabled={loading}
            >
              <Button 
                icon={<UploadOutlined />} 
                size="large" 
                type="primary"
                style={{
                  height: '48px',
                  padding: '0 32px',
                  fontSize: '16px'
                }}
                loading={loading}
              >
                {previewImage ? '重新上传' : '上传植物图片'}
              </Button>
            </Upload>
            
            <div style={{ 
              marginTop: 16, 
              textAlign: 'center',
              color: '#666'
            }}>
              <p>支持JPG、PNG格式，文件小于5MB</p>
              {serviceStatus?.mode === 'demo' && (
                <Alert
                  message="演示模式"
                  description="当前处于演示模式，识别结果仅供参考"
                  type="warning"
                  showIcon
                  style={{ marginTop: 8 }}
                />
              )}
            </div>
          </div>
        </Col>
      </Row>
    </Card>
  );
};

export default Uploader;