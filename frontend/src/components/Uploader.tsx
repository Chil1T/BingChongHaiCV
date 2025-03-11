import React, { useState } from 'react';
import { Upload, message, Button, Spin } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import { uploadImage } from '../api/client';
import './Uploader.module.css';  // 导入CSS模块

interface UploaderProps {
  className?: string;
  testId: string;
  onResult: (result: any, imageUrl: string) => void;
}

const Uploader: React.FC<UploaderProps> = ({ className = 'plant-disease-uploader', testId, onResult }) => {
  const [loading, setLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

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

    // 创建预览URL
    const imageUrl = URL.createObjectURL(file);
    setPreviewUrl(imageUrl);
    
    return true;
  };

  const handleUpload = async (file: File) => {
    try {
      setLoading(true);
      const result = await uploadImage(file);
      
      // 创建图片URL用于显示
      const imageUrl = URL.createObjectURL(file);
      
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

  return (
    <div className={className} data-testid={testId}>
      <Spin spinning={loading}>
        <Upload
          accept="image/*"
          showUploadList={false}
          beforeUpload={beforeUpload}
          customRequest={({ file }) => handleUpload(file as File)}
        >
          <Button icon={<UploadOutlined />} size="large" type="primary" block>
            上传植物图片
          </Button>
        </Upload>
        <div style={{ marginTop: 16, textAlign: 'center' }}>
          <p>支持JPG、PNG格式，文件小于5MB</p>
        </div>
      </Spin>
    </div>
  );
};

export default Uploader;