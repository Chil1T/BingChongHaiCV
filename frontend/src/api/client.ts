import axios, { AxiosError } from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30秒超时
});

export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy';
  mode: 'demo' | 'production';
  model_loaded: boolean;
  loading_in_progress?: boolean;
  numpy_info?: {
    version: string | null;
    compatible: boolean;
    error: string | null;
  };
  model_info?: {
    path: string;
    exists: boolean;
    size_mb: number | null;
    device: string;
  };
  demo_mode_reason?: string;
  error?: string;
}

export interface PredictionResponse {
  status: number;
  data: {
    predictions: Array<{
      class: string;
      probability: number;
    }>;
  };
  inference_time: number;
  demo_mode?: boolean;
  error?: string;
  error_message?: string;
}

export const checkHealth = async (): Promise<HealthCheckResponse> => {
  try {
    const response = await apiClient.get<HealthCheckResponse>('/api/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    return {
      status: 'unhealthy',
      mode: 'demo',
      model_loaded: false,
      error: '服务暂时不可用'
    };
  }
};

export const uploadImage = async (file: File): Promise<PredictionResponse> => {
  const formData = new FormData();
  formData.append('image', file);

  try {
    const response = await apiClient.post<PredictionResponse>('/api/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error uploading image:', error);
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<PredictionResponse>;
      if (axiosError.response?.data) {
        return axiosError.response.data;
      }
      throw new Error(axiosError.message);
    }
    throw new Error('上传图片时发生未知错误');
  }
}; 