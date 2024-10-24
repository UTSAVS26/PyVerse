export interface ChatContent {
    agent: 'user' | 'chatbot';
    message: string;
    loading?: boolean;
}