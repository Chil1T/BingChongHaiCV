import React, { useState } from 'react';
import { Upload, message, Button, Spin } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import { uploadImage } from '../api/client';
import './Uploader.module.css';  // 导入CSS模块

interface UploaderProps {
  className?: string;
  testId: string;
  onResult: (result: any) => void;
}

const Uploader: React.FC<UploaderProps> = ({ className = 'plant-disease-uploader', testId, onResult }) => {
  const [loading, setLoading] = useState(false);

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
    return true;
  };

  const handleUpload = async (file: File) => {
    try {
      setLoading(true);
      const result = await uploadImage(file);
      onResult(result);
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
          <Button icon={<UploadOutlined />} size="large">
            上传植物图片
          </Button>
        </Upload>
      </Spin>
    </div>
  );
};

export default Uploader;