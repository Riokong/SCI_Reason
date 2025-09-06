import json
import re
import os
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline,
    LlavaNextProcessor, LlavaNextForConditionalGeneration
)
from huggingface_hub import login
import torch
from PIL import Image
from typing import Dict, Any

class DualSemanticDistiller:
    def __init__(self, 
                 llama_model: str = "meta-llama/Meta-Llama-3-70B",
                 llava_model: str = "llava-hf/llava-v1.6-mistral-7b-hf",
                 hf_token: str = None):
        """
        Dual Processing: Initialize both Llama (text-only) and LLaVA (vision) models
        """
        self.llama_model_name = llama_model
        self.llava_model_name = llava_model
        
        # Authenticate
        if hf_token:
            login(hf_token)
        
        # Model containers
        self.llama_pipeline = None
        self.llava_processor = None
        self.llava_model = None
        
        self.load_models()
    
    def load_models(self):
        """Load both models with proper error handling"""
        
        print(f"🦙 Loading Llama model: {self.llama_model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.llama_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.llama_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.llama_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("✅ Llama model loaded successfully")
        except Exception as e:
            print(f"❌ Llama loading failed: {e}")
            self.llama_pipeline = None
        
        print(f"👁️ Loading LLaVA model: {self.llava_model_name}")
        try:
            self.llava_processor = LlavaNextProcessor.from_pretrained(self.llava_model_name)
            self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
                self.llava_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            print("✅ LLaVA model loaded successfully")
        except Exception as e:
            print(f"❌ LLaVA loading failed: {e}")
            # Try fallback model
            try:
                print("🔄 Trying fallback LLaVA model...")
                from transformers import LlavaProcessor, LlavaForConditionalGeneration
                self.llava_model_name = "llava-hf/llava-1.5-7b-hf"
                self.llava_processor = LlavaProcessor.from_pretrained(self.llava_model_name)
                self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                    self.llava_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                print("✅ Fallback LLaVA model loaded successfully")
            except Exception as e2:
                print(f"❌ Fallback LLaVA also failed: {e2}")
                self.llava_model = None
    
    def clean_latex(self, text: str) -> str:
        """Clean LaTeX formatting"""
        if not text:
            return ""
        
        patterns = [
            r'\\[a-zA-Z]+\{[^}]*\}', r'\\[a-zA-Z]+', r'\$[^$]*\$',
            r'\\\([^)]*\\\)', r'\\\[[^\]]*\\\]', r'\\&', r'\\%', r'\\_', r'\\#'
        ]
        
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned)
        
        return re.sub(r'\s+', ' ', cleaned).strip()
    
    def generate_llama_insight(self, caption: str, context: str = "") -> str:
        """Generate text-only semantic insight using Llama"""
        
        if not self.llama_pipeline:
            return "Llama model not available"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a scientific literature expert. Extract key insights from figure captions using research context. Respond with 2-3 sentences that capture the main scientific contribution and significance.<|eot_id|><|start_header_id|>user<|end_header_id|>

Research Context:
{context[:1000] if context else "No additional context."}

Figure Caption: {caption}

Provide a concise semantic distillation focusing on the scientific insight, methodology, or finding this figure represents:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

This figure"""
        
        try:
            outputs = self.llama_pipeline(
                prompt,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                return_full_text=False
            )
            
            generated_text = outputs[0]['generated_text'].strip()
            return self.clean_llama_output(generated_text, caption)
            
        except Exception as e:
            return f"Llama processing error: {str(e)[:80]}..."
    
    def generate_llava_insight(self, image_path: str, caption: str, context: str = "") -> str:
        """Generate vision-enhanced semantic insight using LLaVA with proper instruction format"""
        
        if not self.llava_model:
            return "LLaVA model not available"
        
        # Load and validate image
        print(f"   🖼️ Loading image: {image_path}")
        image = self.load_image_safely(image_path)
        if image is None:
            return "Image not found - using text-only fallback"
        
        # Use proper instruction format for Llama-based vision models
        prompt = f"""[INST] <image>
You are a scientific literature expert analyzing research figures. 

Research Context: {context[:800] if context else "No context provided."}

Figure Caption: {caption}

Analyze this figure and provide a concise 2-3 sentence semantic distillation that explains:
1. What the figure visually shows (charts, graphs, data patterns)
2. The key scientific insight or finding it demonstrates
3. How it relates to the research objectives

Focus on combining both visual analysis and scientific interpretation. [/INST]

This figure"""
        
        try:
            inputs = self.llava_processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.llava_model.device)
            
            with torch.no_grad():
                outputs = self.llava_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.llava_processor.tokenizer.eos_token_id
                )
            
            generated_text = self.llava_processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            cleaned_output = self.clean_llava_output(generated_text, caption)
            print(f"   ✅ Generated vision insight: {cleaned_output[:100]}...")
            return cleaned_output
            
        except Exception as e:
            print(f"   ❌ LLaVA processing error: {e}")
            return f"LLaVA processing error: {str(e)[:80]}..."
    
    def load_image_safely(self, image_path: str) -> Image.Image:
        """Load image with detailed error handling and path validation"""
        try:
            # Check if path exists
            if not image_path:
                print(f"   ⚠️ Empty image path provided")
                return None
                
            if not os.path.exists(image_path):
                print(f"   ⚠️ Image file not found: {image_path}")
                # Try alternative paths
                alt_paths = [
                    image_path.replace('images1/', ''),  # Remove images/ prefix
                    f"dataset/{image_path}",            # Add dataset/ prefix
                    f"./{image_path}",                  # Current directory
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        print(f"   ✅ Found image at alternative path: {alt_path}")
                        image_path = alt_path
                        break
                else:
                    return None
            
            # Load and validate image
            image = Image.open(image_path).convert('RGB')
            
            # Check if image is valid (not corrupted)
            if image.size[0] == 0 or image.size[1] == 0:
                print(f"   ⚠️ Invalid image dimensions: {image.size}")
                return None
            
            # Optimize image size for LLaVA
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                print(f"   📏 Resized image from original size to {image.size}")
            
            print(f"   ✅ Successfully loaded image: {image.size}")
            return image
            
        except Exception as e:
            print(f"   ❌ Image loading error for {image_path}: {e}")
            return None
    
    def clean_llama_output(self, generated_text: str, caption: str) -> str:
        """Clean Llama-generated text"""
        
        # Remove Llama-specific artifacts
        artifacts = [
            "Provide a concise", "focusing on", "represents:", "user", 
            "Research Context:", "Figure Caption:", "<|eot_id|>", "<|start_header_id|>",
            "Provide a concise semantic distillation", "methodology, or finding"
        ]
        
        cleaned = generated_text
        for artifact in artifacts:
            cleaned = re.sub(re.escape(artifact), '', cleaned, flags=re.IGNORECASE)
        
        # Remove problematic patterns
        patterns = [
            r'Provide a concise.*?represents.*?:.*?user.*',
            r':.*?user.*',
            r'user.*'
        ]
        
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        return self.extract_meaningful_sentences(cleaned, min_length=20)
    
    def clean_llava_output(self, generated_text: str, caption: str) -> str:
        """Clean LLaVA-generated text (updated for instruction format)"""
        
        # Remove instruction format artifacts
        artifacts = [
            "[INST]", "[/INST]", "<image>", 
            "You are a scientific", "Research Context:", "Figure Caption:",
            "Analyze this figure", "provide a concise", "Focus on combining"
        ]
        
        cleaned = generated_text
        for artifact in artifacts:
            cleaned = re.sub(re.escape(artifact), '', cleaned, flags=re.IGNORECASE)
        
        return self.extract_meaningful_sentences(cleaned, min_length=25)
    
    def extract_meaningful_sentences(self, text: str, min_length: int = 20) -> str:
        """Extract clean, meaningful sentences from generated text"""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Quality filters
            if (len(sentence) >= min_length and 
                sentence.count(' ') >= 3 and
                not any(word in sentence.lower() for word in [
                    'provide', 'analyze', 'focus', 'task', 'concise', 
                    'context:', 'caption:', 'represents:'
                ])):
                meaningful_sentences.append(sentence)
        
        if meaningful_sentences:
            # Take best 2-3 sentences
            result = '. '.join(meaningful_sentences[:3])
            return result + '.' if not result.endswith('.') else result
        else:
            # Fallback: clean and truncate
            cleaned = re.sub(r'^[^A-Z]*', '', text)  # Remove leading junk
            return cleaned[:150] + '...' if len(cleaned) > 150 else cleaned
    
    def process_jsonl_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process single record with dual analysis"""
        
        # Extract record data
        paper_id = record.get('paper_id', 'unknown')
        figure_id = record.get('figure_id', 'fig1')
        figure_path = record.get('figure_path', '')
        caption = record.get('caption', '')
        
        # Build rich context
        methodology = record.get('methodology', '')
        results = record.get('results', '')
        conclusion = record.get('conclusion', '')
        introduction = record.get('introduction', '')
        
        context_parts = []
        if methodology:
            context_parts.append(f"Methodology: {methodology[:400]}...")
        if results:
            context_parts.append(f"Results: {results[:400]}...")
        if conclusion:
            context_parts.append(f"Conclusion: {conclusion[:200]}...")
        
        rich_context = "\n\n".join(context_parts)
        
        # Generate insights from both models
        print(f"🔄 Processing {paper_id} - {figure_id}")
        
        llama_insight = self.generate_llama_insight(caption, rich_context)
        llava_insight = self.generate_llava_insight(figure_path, caption, rich_context)
        
        # Create dual-processed record
        result = {
            'paper_id': paper_id,
            'figure_id': figure_id,
            'figure_path': figure_path,
            'caption': self.clean_latex(caption),
            
            # Dual insights
            'llama_insight': llama_insight,
            'llava_insight': llava_insight,
            
            # Analysis metadata
            'dual_analysis': {
                'text_only_analysis': llama_insight,
                'vision_enhanced_analysis': llava_insight,
                'image_available': os.path.exists(figure_path),
                'image_successfully_processed': not any(phrase in llava_insight.lower() for phrase in [
                    "image not found", "not available", "processing error", "unreadable"
                ]),
                'llama_model': self.llama_model_name,
                'llava_model': self.llava_model_name,
                'both_models_successful': (
                    not any(phrase in llama_insight.lower() for phrase in ["error", "not available"]) and 
                    not any(phrase in llava_insight.lower() for phrase in [
                        "error", "not available", "image not found", "unreadable"
                    ])
                ),
                'analysis_quality': {
                    'llama_success': not any(phrase in llama_insight.lower() for phrase in ["error", "not available"]),
                    'llava_success': not any(phrase in llava_insight.lower() for phrase in [
                        "error", "not available", "image not found", "unreadable"
                    ]),
                    'has_vision_data': os.path.exists(figure_path)
                }
            }
        }
        
        return result
    
    def process_jsonl_file(self, input_file: str, output_file: str):
        """Process entire JSONL file with dual analysis"""
        
        # Count total records
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        print(f"📊 Processing {total_lines} records with dual analysis...")
        
        all_results = []
        successful_dual = 0
        llama_only = 0
        llava_only = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Dual processing"), 1):
                try:
                    record = json.loads(line.strip())
                    result = self.process_jsonl_record(record)
                    all_results.append(result)
                    
                    # Track success rates
                    if result['dual_analysis']['both_models_successful']:
                        successful_dual += 1
                    elif "error" not in result['llama_insight'].lower():
                        llama_only += 1
                    elif "error" not in result['llava_insight'].lower():
                        llava_only += 1
                    
                    # Save progress every 3 records (dual processing is slower)
                    if line_num % 3 == 0:
                        with open(output_file, 'w', encoding='utf-8') as out_f:
                            for res in all_results:
                                out_f.write(json.dumps(res, ensure_ascii=False) + '\n')
                
                except Exception as e:
                    print(f"\n❌ Error processing record {line_num}: {e}")
        
        # Final save
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Report results
        print(f"\n✅ Dual processing complete!")
        print(f"📁 Saved {len(all_results)} records to {output_file}")
        print(f"🎯 Success rates:")
        print(f"   Both models successful: {successful_dual}")
        print(f"   Llama only: {llama_only}")
        print(f"   LLaVA only: {llava_only}")
        print(f"   Success rate: {successful_dual/len(all_results)*100:.1f}%")
        
        return all_results

# Main execution function
def main():
    # Authentication
    login("####")
    
    # Configuration
    LLAMA_MODEL = "meta-llama/Meta-Llama-3-70B"
    LLAVA_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"
    INPUT_FILE = "dataset/finance_dataset5.jsonl"
    OUTPUT_FILE = "dataset/finance_dual_distilled.jsonl"
    
    # Initialize dual processor
    print("🚀 Initializing Dual Semantic Distiller...")
    distiller = DualSemanticDistiller(
        llama_model=LLAMA_MODEL,
        llava_model=LLAVA_MODEL
    )
    
    # Process with both models
    print(f"📂 Processing {INPUT_FILE} with dual analysis...")
    results = distiller.process_jsonl_file(INPUT_FILE, OUTPUT_FILE)
    
    # Display sample results
    print("\n🔍 Sample dual-analyzed results:")
    for i, result in enumerate(results[:2]):
        print(f"\n--- Sample {i+1} ---")
        print(f"📄 Paper: {result['paper_id']} - {result['figure_id']}")
        print(f"📝 Caption: {result['caption']}")
        print(f"\n🦙 Llama Analysis:")
        print(f"   {result['llama_insight']}")
        print(f"\n👁️ LLaVA Analysis:")
        print(f"   {result['llava_insight']}")
        print(f"\n📊 Metadata:")
        print(f"   Image available: {result['dual_analysis']['image_available']}")
        print(f"   Both successful: {result['dual_analysis']['both_models_successful']}")
        print("=" * 80)

if __name__ == "__main__":
    main()