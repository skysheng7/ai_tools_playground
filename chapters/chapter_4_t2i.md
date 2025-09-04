# 🎨 Chapter 4: Text-to-Image (T2I) Models

*Note: All GenAI tools & models demonstrated below are NOT recommended by UBC due to privacy and safety reasons (but you are NOT restricted from using them)*


Text-to-Image (T2I) models have revolutionized the way we create and edit visual content by enabling users to generate images from textual descriptions. These AI systems can transform written prompts into stunning visual artwork, photographs, and creative designs.

---

## 1. Leading T2I Models

### 1.1 OpenAI Models

#### GPT-5 
- State-of-the-art T2I model for image generation and image editing
- **Access**: Available through [ChatGPT](https://chatgpt.com/) Plus subscription ($20/month)

#### GPT-4o
- **Access**: Available through [ChatGPT](https://chatgpt.com/) Plus subscription ($20/month)
- **Strengths**: Excellent prompt following and realistic image generation

Here are some example output images from GPT-4o:

```{image} ../images/gpt5.png
:alt: gpt5
:width: 100%
```

### 1.2 Google's Nano Banana (Gemini Flash 2.5)

**💡 What is Nano Banana?**
[Nano Banana](https://aistudio.google.com/?model=gemini-2.5-flash-image-preview) is Google's latest AI image editing model, also known as **Gemini-2.5-flash-image-preview**, has recently gone viral for its advanced image editing capabilities.

**🌟 Key Features:**
- **Character Consistency**: Maintains the same character across multiple scenes
- **Advanced Editing**: Blur backgrounds, remove objects, modify poses
- **Colorization**: Transform black-and-white photos into color
- **Multi-image Combination**: Seamlessly blend multiple images
- **Free Access**

Here are some example output images from Nano Banana:

```{image} ../images/banana1.png
:alt: banana 1
:width: 100%
```

```{image} ../images/banana2.png
:alt: banana 2
:width: 100%
```


### 1.3 Midjourney

**💡 What is Midjourney?**
[Midjourney](https://midjourney.com) is a  popular AI program that generates images from textual descriptions, known for its artistic and creative outputs.

**🎨 Key Strengths:**
- **Artistic Style**: Exceptional at creating artistic and stylized images
- **Community**: Large community of digital artists and designers
- **Quality**: High-quality, creative outputs, with wide range of artistic styles
- Perfect for digital art creation, concept art and design

**Access**: 
- **Platform**: Discord-based interface
- **Pricing**: Subscription-based ($10-60/month)

Here are some example output images from Midjourney:

```{image} ../images/midjourney.png
:alt: midjourney
:width: 100%
```

### 1.4 Stable Diffusion

**💡 What is Stable Diffusion?**
[Stable Diffusion](https://stability.ai) is an open-source AI model capable of generating high-quality images from text prompts, offering flexibility and control.

**🔓 Key Advantages:**
- **Open Source**: Free to use and modify
- **Local Running**: Can run on your own hardware
- **Customization**: Fine-tune models for specific needs
- **Control**: Advanced parameters for precise control
- **Community**: Large open-source community and extensions
- Perfect for developers, customized model training, local image generation


Here are some example output images from Stable Diffusion:

```{image} ../images/sd.png
:alt: stable diffusion
:width: 100%
```

---

## 2. Try Them All for Free: T2I Arena

### 2.1 What is T2I Arena?

[**T2I Arena**](https://lmarena.ai/?chat-modality=image) is a battle field for different text-to-image generative models, similar to the Chatbot Arena for language models.

**🆓 Free Access Features:**
- **Try All Models**: Access to various T2I models including DALL-E 3, Stable Diffusion, and more
- **Side-by-Side Comparison**: Compare outputs from different models with the same prompt
- **Vote & Contribute**: Help rank models by voting on which outputs you prefer
- **No Subscription Required**: Completely free to use

**🎯 How It Works:**
1. Enter your text prompt
2. Two anonymous models generate images
3. Vote for the better result
4. Model names are revealed after voting
5. Your votes contribute to the global rankings

---

## 3. Choosing the Right T2I Model

### 3.1 Model Comparison

| Model | Best For | Cost | Access |
|-------|----------|------|--------|
| **GPT-4o/GPT-5** | General use, realistic images | $20/month | ChatGPT Plus |
| **Nano Banana** | Photo editing, character consistency | Free | Google AI Studio |
| **Midjourney** | Artistic creation, creative projects | $10-60/month | Discord |
| **Stable Diffusion** | Customization, local use | Free | Various platforms |

### 3.2 Workflow Recommendations

**For Beginners**: Start with **T2I Arena** to explore different models for free

**For Artists**: **Midjourney** for creative and artistic projects

**For Developers**: **Stable Diffusion** for customization and local deployment

**For Photo Editing**: **Nano Banana** for quick edits and enhancements

**For Presentation**: **GPT-4o** or **GPT-5** if you're already using ChatGPT Plus

---

## 4. Tips for prompting for images

**Be Specific!** "A red sports car" is less effective than prompting "A bright red Ferrari 488 GTB parked on a mountain road at sunset"

### 🧙‍♀️ Here is my usual workflow:

1. 💭 **Use LLM to describe image:** Use LLM (e.g., Claude, ChatGPT) to first generate a few ideas, or create a vivid description of the image you wish to generate

2. 🎨 (Optional) **Include Style**: Add artistic style descriptions like "in the style of Van Gogh" or "photorealistic". Here is a [guide for artistic style](https://zapier.com/blog/ai-art-styles/).

3. 🔅 (Optional) **Specify Details**: Mention lighting, composition, color theme, and mood

4. 🪄 Use the well-crafted text description to prompt one of the T2I model for image generation

5. 🎲 (Optional) **Experiment**: Try the same prompt multiple times to see variations

6. 🤩 If there are small details or words that you wish to change in the image, use Google's [**Nano Banana**](https://aistudio.google.com/?model=gemini-2.5-flash-image-preview) to edit the generated image for final perfection (adjust the temperature as needed, more randomness if temperature is closer to 1).

---

## 5. Ethical Considerations

### 5.1 Important Reminders

⚠️ **Lack of Consent from Artist**: Most of the training data used for developing these T2I models are images automatically scapped from the internet, without the consent from original artists who created them!

⚠️ **Copyright and Fair Use**: Be mindful of generating images in copyrighted styles or of copyrighted characters

⚠️ **Commercial Use**: Check licensing terms before using generated images commercially

### 4.2 Responsible AI Art Creation

- **Machine Unlearning**: Contribute to the [Machine Unlearning](https://www.ibm.com/think/insights/machine-unlearning) work and research where we can teach model to "forget" about some of its training data, if some artists or creators do not want their work to be used in training genAI
- **Respect Artists**: Don't try to replicate living artists' exact styles without permission
- **Be Transparent**: Disclose when content is AI-generated when appropriate
- **Avoid Harmful Content**: Don't create images that could harm or mislead others
- **Support Human Artists**: AI should complement, not replace, human creativity
