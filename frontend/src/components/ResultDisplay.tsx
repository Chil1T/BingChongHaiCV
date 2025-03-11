import React from 'react';
import { Card, Typography, Tag, Progress, Divider, Descriptions, Empty, Alert } from 'antd';
import { CheckCircleOutlined, WarningOutlined, InfoCircleOutlined } from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;

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

interface ResultDisplayProps {
  result: Result | null;
}

// 疾病信息数据库
const diseaseInfo: Record<string, { description: string; treatment: string; severity: 'low' | 'medium' | 'high' }> = {
  'Apple___Apple_scab': {
    description: '苹果黑星病是由Venturia inaequalis真菌引起的常见苹果病害，表现为叶片和果实上的橄榄色至黑色斑点。',
    treatment: '使用杀菌剂喷洒，清除落叶，选择抗病品种。',
    severity: 'high'
  },
  'Apple___Black_rot': {
    description: '苹果黑腐病是由Botryosphaeria obtusa真菌引起的，会导致叶片、果实和树枝腐烂。',
    treatment: '修剪受感染的枝条，使用杀菌剂，保持果园卫生。',
    severity: 'high'
  },
  'Apple___Cedar_apple_rust': {
    description: '苹果锈病是由Gymnosporangium juniperi-virginianae真菌引起的，在叶片上形成橙色斑点。',
    treatment: '移除附近的杜松树，使用杀菌剂，选择抗病品种。',
    severity: 'medium'
  },
  'Apple___healthy': {
    description: '健康的苹果植株，无明显病害症状。',
    treatment: '继续保持良好的园艺实践，定期监测。',
    severity: 'low'
  },
  'Tomato___Early_blight': {
    description: '番茄早疫病是由Alternaria solani真菌引起的，表现为叶片上的同心圆斑点。',
    treatment: '避免叶面浇水，使用杀菌剂，轮作。',
    severity: 'medium'
  },
  'Tomato___Late_blight': {
    description: '番茄晚疫病是由Phytophthora infestans卵菌引起的，会导致叶片、茎和果实腐烂。',
    treatment: '使用杀菌剂，避免过度浇水，确保良好通风。',
    severity: 'high'
  },
  'Tomato___healthy': {
    description: '健康的番茄植株，无明显病害症状。',
    treatment: '继续保持良好的园艺实践，定期监测。',
    severity: 'low'
  }
};

// 默认疾病信息
const defaultDiseaseInfo = {
  description: '该植物疾病是由病原体引起的，可能会影响植物的生长和产量。',
  treatment: '建议咨询当地农业专家获取具体的治疗方案。',
  severity: 'medium' as const
};

const ResultDisplay: React.FC<ResultDisplayProps> = ({ result }) => {
  if (!result) {
    return (
      <Card title="识别结果" bordered={false}>
        <Empty description="请上传图片进行识别" />
      </Card>
    );
  }

  if (result.status !== 200 || result.error) {
    return (
      <Card title="识别结果" bordered={false}>
        <Alert
          message="识别失败"
          description={result.error || '服务器处理请求时出错'}
          type="error"
          showIcon
        />
      </Card>
    );
  }

  const topPrediction = result.data.predictions[0];
  const diseaseData = diseaseInfo[topPrediction.class] || defaultDiseaseInfo;

  // 获取疾病的颜色标签
  const getDiseaseTagColor = (className: string): string => {
    if (className.includes('healthy')) return 'success';
    if (className.includes('blight') || className.includes('rot')) return 'error';
    if (className.includes('spot') || className.includes('scorch')) return 'warning';
    if (className.includes('mildew') || className.includes('rust')) return 'processing';
    return 'default';
  };

  // 格式化类名显示
  const formatClassName = (className: string): string => {
    return className.replace(/_/g, ' ').replace(/___/g, ' - ');
  };

  // 获取严重程度图标
  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'low':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'medium':
        return <WarningOutlined style={{ color: '#faad14' }} />;
      case 'high':
        return <InfoCircleOutlined style={{ color: '#f5222d' }} />;
      default:
        return <InfoCircleOutlined />;
    }
  };

  return (
    <Card title="识别结果" bordered={false}>
      <Paragraph>
        <Text strong>推理时间: </Text>
        <Text>{result.inference_time} 毫秒</Text>
      </Paragraph>

      <Divider orientation="left">预测结果</Divider>

      {result.data.predictions.map((prediction, index) => (
        <div key={index} style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <Tag color={getDiseaseTagColor(prediction.class)}>
              {formatClassName(prediction.class)}
            </Tag>
            <Text strong>{prediction.probability.toFixed(2)}%</Text>
          </div>
          <Progress
            percent={prediction.probability}
            status={index === 0 ? "active" : "normal"}
            strokeColor={index === 0 ? "#52c41a" : "#1890ff"}
          />
        </div>
      ))}

      {topPrediction.probability > 50 && (
        <>
          <Divider orientation="left">疾病详情</Divider>
          <Descriptions bordered size="small" column={1}>
            <Descriptions.Item 
              label={
                <span>
                  {getSeverityIcon(diseaseData.severity)} 严重程度
                </span>
              }
            >
              {diseaseData.severity === 'low' ? '低' : diseaseData.severity === 'medium' ? '中' : '高'}
            </Descriptions.Item>
            <Descriptions.Item label="描述">{diseaseData.description}</Descriptions.Item>
            <Descriptions.Item label="建议治疗">{diseaseData.treatment}</Descriptions.Item>
          </Descriptions>
        </>
      )}
    </Card>
  );
};

export default ResultDisplay; 