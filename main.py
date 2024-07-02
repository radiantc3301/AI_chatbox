import pygame
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("AI Chatbot")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# Fonts
font = pygame.font.Font(None, 32)

# Chat history
chat_history = []

# Input box
input_box = pygame.Rect(10, height - 50, width - 20, 40)
input_text = ""

# Load AI model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def get_ai_response(input_text):
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if input_text:
                    chat_history.append(("You: " + input_text))
                    ai_response = get_ai_response(input_text)
                    chat_history.append(("AI: " + ai_response))
                    input_text = ""
            elif event.key == pygame.K_BACKSPACE:
                input_text = input_text[:-1]
            else:
                input_text += event.unicode

    # Fill the background
    screen.fill(WHITE)

    # Draw input box
    pygame.draw.rect(screen, GRAY, input_box)
    text_surface = font.render(input_text, True, BLACK)
    screen.blit(text_surface, (input_box.x + 5, input_box.y + 5))

    # Display chat history
    y_offset = 10
    for message in chat_history[-10:]:  # Show last 10 messages
        text_surface = font.render(message, True, BLACK)
        screen.blit(text_surface, (10, y_offset))
        y_offset += 40

    pygame.display.flip()

pygame.quit()