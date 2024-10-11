import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ChatContent } from './chat-content.interface';

@Injectable({
  providedIn: 'root',
})
export class ChatService {
  constructor(private httpClient: HttpClient) { }

  chat(chatContent: ChatContent): Observable<ChatContent> {
    return this.httpClient.post<ChatContent>('http://localhost:3000/api/chatbot', chatContent);
  }
}