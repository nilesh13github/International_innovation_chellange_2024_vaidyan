from flask import Flask, render_template, request
from flask_socketio import SocketIO, send
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'
socketio = SocketIO(app)

# Route to serve the chat page
@app.route('/')
def index():
    return render_template('chat.html')

# Handle message sending
@socketio.on('message')
def handle_message(msg):
    print(f"Received message: {msg}")
    
    # Broadcast the message to all connected clients
    send(msg, broadcast=True)
    
    # Send a reply from the backend after a short delay
    time.sleep(1)  # Delay to make the reply feel more natural
    automated_reply = f"Server: Received your message - '{msg}'"
    send(automated_reply, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, debug=True)
