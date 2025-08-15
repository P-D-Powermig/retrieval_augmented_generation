css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color:rgb(37, 5, 0)
}
.chat-message.bot {
    background-color:rgb(0, 16, 35)
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 100px;
  max-height: 100px;
  border-radius: 30%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.postimg.cc/X7Jr2PvY/GIU-AMA-255-07.jpg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.postimg.cc/L8MLBL79/Chat-GPT-Image-10-de-abr-de-2025-17-41-46.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''