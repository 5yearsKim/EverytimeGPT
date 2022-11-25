import axios from 'axios';
import { SERVER_URL } from '../config';


export const server = axios.create({
  baseURL: SERVER_URL,
});

export async function generateText(sent, keywords=[]) {
  const body = {
    sent: sent,
    num_sent: 3,
  };
  if (keywords) {
    body['keywords'] = keywords;
  }
  const rsp = await server.post('/generate_sample', body);
  // console.log(rsp);
  return rsp['data']['generated'];
}