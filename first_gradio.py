import gradio as gr
def greet(name, intensity, button):
  return "Hello, " + name + "!" * int(intensity) + button
demo = gr.Interface(
  fn=greet,
  inputs=["text", "slider", "text"],
  outputs=["text"],
)
demo.launch(server_name="127.0.0.1", server_port= 7860)