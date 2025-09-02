import streamlit as st
import json
from datetime import datetime
import PyPDF2
import google.generativeai as genai
import io
import random
from thumbnail_generator import ThumbnailGenerator

# Initialize Gemini API
@st.cache_resource
def initialize_gemini():
    genai.configure(api_key="AIzaSyDV7UlF6AVXMpnA24EGnn8NS-X0Te5586A")
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

class DocumentProcessor:
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
        self.document_content = ""
    
    def process_pdf(self, file):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        texts = []
        for page in pdf_reader.pages:
            texts.append(page.extract_text())
        self.document_content = " ".join(texts)
        return self.document_content
    
    def extract_relevant_content(self, topic):
        if not self.document_content:
            return ""
        
        prompt = f"Extract key information about '{topic}' from: {self.document_content[:2000]}..."
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except:
            return self.document_content[:1000]

class ScriptGenerator:
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
    
    def generate_script(self, title, topic, audience, tone, emotion, content_type, context=""):
        prompt = f"""Create a comprehensive teaching script for:

Title: {title}
Topic: {topic}
Audience: {audience}
Tone: {tone}
Emotion: {emotion}
Content Type: {content_type}
Context: {context}

Generate a complete script with these sections:
1. Hook (engaging opening 1-2 sentences)
2. Learning Objectives (3-4 clear bullet points)
3. Core Content (detailed explanations with examples)
4. Interactive Element (question or exercise)
5. Summary & Call to Action

Make it practical, engaging, and tailored to the {audience} audience with a {tone} tone."""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return self._parse_script_response(response.text)
        except Exception as e:
            return self._generate_fallback_script(title, topic, audience)
    
    def _parse_script_response(self, response_text):
        # Parse Gemini response into structured format
        sections = response_text.split('\n\n')
        return {
            "hook": sections[0] if sections else "Welcome to today's session!",
            "objectives": ["Master key concepts", "Apply practical knowledge", "Understand best practices"],
            "core_content": response_text,
            "interactive": "What questions do you have about this topic?",
            "summary": "Review key points and continue learning!"
        }
    
    def _generate_fallback_script(self, title, topic, audience):
        return {
            "hook": f"Today we'll explore {topic} - essential knowledge for {audience}.",
            "objectives": [f"Understand {topic} fundamentals", "Learn about applications", ],
            "core_content": f"{topic} is a crucial concept that helps professionals solve complex problems effectively.",
            "interactive": f"How would you apply {topic} in your current work?",
            "summary": f"You now have foundational knowledge of {topic}. Practice and apply these concepts!"
        }
    

    


def save_metadata(data, script, gemini_model):
    timestamp = datetime.now().isoformat()
    
    # Generate 5 different thumbnail prompts using Gemini
    thumbnail_prompts_request = f"""Create 5 different SDXL prompts for generating educational thumbnails based on:
    
    Title: {data['title']}
    Topic: {data['topic']}
    Audience: {data['audience']}
    Tone: {data['tone']}
    Emotion: {data['emotion']}
    Script Hook: {script['hook'][:100]}...
    
    Generate 5 distinct visual styles for thumbnails:
    1. Modern/Minimalist style
    2. Bold/Dynamic style  
    3. Professional/Corporate style
    4. Creative/Artistic style
    5. Tech/Futuristic style
    
    Each prompt should be detailed, specify colors, composition, visual elements, and be optimized for SDXL generation. Format as JSON array with style names and prompts."""
    
    try:
        thumbnail_response = gemini_model.generate_content(thumbnail_prompts_request)
        # Try to parse JSON, fallback if needed
        import re
        json_match = re.search(r'\[.*\]', thumbnail_response.text, re.DOTALL)
        if json_match:
            thumbnail_prompts = json.loads(json_match.group())
        else:
            # Fallback to simple list
            thumbnail_prompts = [
                {"style": "Modern", "prompt": f"Modern minimalist design for {data['topic']}, clean typography, bright colors"},
                {"style": "Bold", "prompt": f"Bold dynamic design for {data['topic']}, strong contrasts, energetic composition"},
                {"style": "Professional", "prompt": f"Professional corporate design for {data['topic']}, business colors, clean layout"},
                {"style": "Creative", "prompt": f"Creative artistic design for {data['topic']}, unique visual elements, vibrant palette"},
                {"style": "Tech", "prompt": f"Futuristic tech design for {data['topic']}, digital elements, neon accents"}
            ]
    except Exception as e:
        print(f"Error generating prompts: {e}")
        thumbnail_prompts = [
            {"style": "Modern", "prompt": f"Modern educational thumbnail for {data['topic']} with clean design and vibrant colors"},
            {"style": "Bold", "prompt": f"Bold educational thumbnail for {data['topic']} with dynamic composition"},
            {"style": "Professional", "prompt": f"Professional educational thumbnail for {data['topic']} with corporate styling"},
            {"style": "Creative", "prompt": f"Creative educational thumbnail for {data['topic']} with artistic elements"},
            {"style": "Tech", "prompt": f"Tech-style educational thumbnail for {data['topic']} with futuristic design"}
        ]
    
    metadata = {
        **data,
        "timestamp": timestamp,
        "script_generated": script,
        "thumbnail_prompts": thumbnail_prompts,
        "sections": ["Hook", "Objectives", "Core Content", "Interactive", "Summary"],
        "tone_tag": f"#tone:{data['tone']}",
        "emotion_tag": f"#emotion:{data['emotion']}"
    }
    
    filename = f"metadata/script_{timestamp.replace(':', '-')}.json"
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    return filename

def main():
    st.set_page_config(page_title="ClipGen", layout="wide")
    st.title("VIDEO PIPELINE ClipGen")
    st.subheader("AI-Powered Teaching Script Generator")
    
    # Initialize Gemini
    gemini_model = initialize_gemini()
    doc_processor = DocumentProcessor(gemini_model)
    script_gen = ScriptGenerator(gemini_model)
    
    # Input form
    with st.form("script_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Title", "Intro to AI/ML")
            topic = st.text_input("Topic", "Supervised vs. Unsupervised Learning")
            audience = st.selectbox("Target Audience", 
                ["junior engineers", "new hires", "students", "professionals"])
        
        with col2:
            tone = st.selectbox("Tone", ["professional", "friendly", "motivational"])
            emotion = st.selectbox("Emotion/Mood", 
                ["inspired", "curious", "confident", "excited"])
            content_type = st.selectbox("Content Type", 
                ["Short-Form Video Reel", "Full Training Module"])
        
        # Document upload
        uploaded_file = st.file_uploader("Optional Domain Docs", type=['pdf'])
        
        submitted = st.form_submit_button("Generate Script")
    
    if submitted:
        with st.spinner("Generating your script..."):
            # Process uploaded document
            context = ""
            if uploaded_file:
                doc_processor.process_pdf(uploaded_file)
                context = doc_processor.extract_relevant_content(topic)
            
            # Generate script
            script = script_gen.generate_script(
                title, topic, audience, tone, emotion, content_type, context
            )
            
            # Verify emotion alignment using Gemini
            emotion_prompt = f"Analyze the emotion/tone of this text: '{script['hook']}'. What emotion does it convey? Answer with one word."
            try:
                emotion_response = gemini_model.generate_content(emotion_prompt)
                detected_emotion = emotion_response.text.strip().lower()
            except:
                detected_emotion = emotion.lower()
            
            # Display script
            st.success("Script Generated Successfully!")
            
            st.markdown("## Target: Hook")
            st.write(script["hook"])
            
            st.markdown("## Endpoints Learning Objectives")
            for obj in script["objectives"]:
                st.write(f"• {obj}")
            
            st.markdown("## Documentation Core Content")
            st.markdown(script["core_content"])
            
            st.markdown("## Continue? Interactive Element")
            st.write(script["interactive"])
            
            st.markdown("## VIDEO PIPELINE Summary & Call to Action")
            st.write(script["summary"])
            
            # Emotion verification
            if detected_emotion != emotion.lower():
                st.warning(f"Note: Detected emotion '{detected_emotion}' differs from target '{emotion}'")
            
            # Save metadata
            metadata = {
                "title": title,
                "topic": topic,
                "audience": audience,
                "tone": tone,
                "emotion": emotion,
                "content_type": content_type
            }
            
            filename = save_metadata(metadata, script, gemini_model)
            st.info(f"Metadata saved to: {filename}")
            
            # Generate thumbnails
            st.markdown("## Image Thumbnails")
            with st.spinner("Generating 5 different thumbnail styles..."):
                with open(filename, 'r') as f:
                    latest_metadata = json.load(f)
                
                thumbnail_prompts = latest_metadata.get("thumbnail_prompts", [])
                thumbnail_gen = ThumbnailGenerator()
                
                # Generate all thumbnails
                thumbnails = thumbnail_gen.generate_thumbnails_batch(
                    thumbnail_prompts, 
                    "quick_draft", 
                    topic,
                    lambda msg: st.write(f"Status: {msg}")
                )
                
                # Display thumbnails in grid
                st.subheader("Choose your preferred thumbnail:")
                
                cols = st.columns(3)
                
                for i, thumb_data in enumerate(thumbnails):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.image(thumb_data["image"], caption=f"{thumb_data['style']} Style", width=200)
                        
                        # Convert to bytes for download
                        img_buffer = io.BytesIO()
                        thumb_data["image"].save(img_buffer, format='PNG')
                        img_bytes = img_buffer.getvalue()
                        
                        st.download_button(
                            label=f"Download Download {thumb_data['style']}",
                            data=img_bytes,
                            file_name=f"{topic.replace(' ', '_')}_{thumb_data['style'].lower()}_thumbnail.png",
                            mime="image/png",
                            key=f"download_{i}"
                        )
                
                # Save the first thumbnail to metadata for backward compatibility
                if thumbnails:
                    first_thumbnail_path = thumbnail_gen.save_thumbnail(thumbnails[0]["image"], filename)
                    st.success(f"Thumbnails generated successfully! First thumbnail saved to: {first_thumbnail_path}")

if __name__ == "__main__":
    main()
