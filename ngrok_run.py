from pyngrok import ngrok

# open tunnel to port 5000
public_url = ngrok.connect(5000)

print("🔥 Public URL:", public_url)