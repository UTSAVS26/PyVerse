import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { ChatService } from './chat.service';
import { ChatContent } from './chat-content.interface';
import { LineBreakPipe } from './line-break.pipe';
import { finalize } from 'rxjs';

@Component({
  selector: 'corp-root',
  standalone: true,
  imports: [
    CommonModule,
    MatIconModule,
    MatCardModule,
    MatInputModule,
    MatButtonModule,
    MatFormFieldModule,
    FormsModule,
    LineBreakPipe,
  ],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent {
  message = '';

  contents: ChatContent[] = [];

  constructor(private chatService: ChatService) {}

  sendMessage(message: string): void {
    const chatContent: ChatContent = {
      agent: 'user',
      message,
    };

    this.contents.push(chatContent);
    this.contents.push({
      agent: 'chatbot',
      message: '...',
      loading: true,
    });
    
    this.message = '';
    this.chatService
      .chat(chatContent)
      .pipe(
        finalize(() => {
          const loadingMessageIndex = this.contents.findIndex(
            (content) => content.loading
          );
          if (loadingMessageIndex !== -1) {
            this.contents.splice(loadingMessageIndex, 1);
          }
        })
      )
      .subscribe((content) => {
        this.contents.push(content);
      });
  }
}