import axios, { AxiosInstance } from 'axios';

const API_URL = 'http://localhost:8000';  // adjust this to your backend URL

export const api: AxiosInstance = axios.create({
  baseURL: API_URL,
});

export const getTokens = async (): Promise<string[]> => {
  const response = await api.get<string[]>('/tokens');
  return response.data;
};

// Add more API calls as needed