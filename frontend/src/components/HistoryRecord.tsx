import React, { useState } from 'react';
import { Modal, Button, List, Tag, Image, Typography, Badge } from 'antd';
import { HistoryOutlined } from '@ant-design/icons';
import { PredictionResponse } from '../api/client';

const { Text } = Typography;

interface HistoryItem {
  id: string;
  timestamp: string;
  imageUrl: string;
  result: PredictionResponse;
}

interface HistoryRecordProps {
  className?: string;
  records: HistoryItem[];
}

const HistoryRecord: React.FC<HistoryRecordProps> = ({ className, records }) => {
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<HistoryItem | null>(null);
  const [detailModalVisible, setDetailModalVisible] = useState(false);

  const showModal = () => setIsModalVisible(true);
  const handleCancel = () => setIsModalVisible(false);

  const showDetailModal = (record: HistoryItem) => {
    setSelectedRecord(record);
    setDetailModalVisible(true);
  };

  const handleDetailCancel = () => {
    setDetailModalVisible(false);
    setSelectedRecord(null);
  };

  return (
    <>
      <Badge count={records.length}>
        <Button
          type="link"
          icon={<HistoryOutlined />}
          onClick={showModal}
          className={className}
        >
          识别历史
        </Button>
      </Badge>

      <Modal
        title="识别历史记录"
        open={isModalVisible}
        onCancel={handleCancel}
        footer={null}
        width={600}
      >
        <List
          itemLayout="horizontal"
          dataSource={records}
          renderItem={(record) => (
            <List.Item
              key={record.id}
              actions={[
                <Button type="link" onClick={() => showDetailModal(record)}>
                  查看详情
                </Button>
              ]}
            >
              <List.Item.Meta
                avatar={
                  <Image
                    src={record.imageUrl}
                    width={50}
                    height={50}
                    style={{ objectFit: 'cover' }}
                  />
                }
                title={record.result.data.predictions[0].class}
                description={`识别时间: ${record.timestamp}`}
              />
              <div>
                <Tag color="blue">
                  {record.result.data.predictions[0].probability.toFixed(1)}%
                </Tag>
              </div>
            </List.Item>
          )}
        />
      </Modal>

      <Modal
        title="识别详情"
        open={detailModalVisible}
        onCancel={handleDetailCancel}
        footer={null}
        width={800}
      >
        {selectedRecord && (
          <div>
            <div style={{ textAlign: 'center', marginBottom: 16 }}>
              <Image
                src={selectedRecord.imageUrl}
                style={{ maxWidth: '100%', maxHeight: 400 }}
              />
            </div>
            
            <div style={{ marginBottom: 16 }}>
              <Text strong>识别时间：</Text>
              <Text>{selectedRecord.timestamp}</Text>
            </div>
            
            <div style={{ marginBottom: 16 }}>
              <Text strong>推理时间：</Text>
              <Text>{selectedRecord.result.inference_time}ms</Text>
            </div>
            
            <List
              header={<div><Text strong>识别结果</Text></div>}
              bordered
              dataSource={selectedRecord.result.data.predictions}
              renderItem={(prediction, index) => (
                <List.Item>
                  <List.Item.Meta
                    title={prediction.class}
                    description={
                      <div style={{ width: '100%' }}>
                        <div style={{ marginBottom: 8 }}>
                          <Tag color={index === 0 ? 'green' : 'blue'}>
                            {prediction.probability.toFixed(1)}%
                          </Tag>
                        </div>
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </div>
        )}
      </Modal>
    </>
  );
};

export default HistoryRecord; 