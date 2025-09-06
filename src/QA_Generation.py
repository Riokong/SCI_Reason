import spacy
import re
from typing import Dict, List, Tuple, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json
import torch
import spacy
import re
from typing import Dict, List, Tuple, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json
import torch

class ContextualGapCreator:
    def __init__(self):
        # Load NLP models
        self.nlp = spacy.load("en_core_web_sm")
        self.setup_custom_patterns()
        self.setup_ml_models()
        
        # Reasoning type configurations
        self.reasoning_configs = {
            "professional_entity_location": {"min_masks": 1, "max_masks": 2, "threshold": 5},
            "multimodal_temporal_reasoning": {"min_masks": 1, "max_masks": 3, "threshold": 8},
            "cross_subgraph_role_reasoning": {"min_masks": 2, "max_masks": 4, "threshold": 10},
            "causal_mechanism_reasoning": {"min_masks": 1, "max_masks": 2, "threshold": 6},
            "methodological_technical_reasoning": {"min_masks": 1, "max_masks": 3, "threshold": 7}
        }
        
    def setup_custom_patterns(self):
        """Add domain-agnostic NER patterns that work across academic fields"""
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        
        patterns = [
            # Dataset patterns (generic across domains)
            {"label": "DATASET", "pattern": [{"TEXT": {"REGEX": r"[A-Z][a-zA-Z0-9-]*"}}, {"LOWER": {"IN": ["dataset", "corpus", "database", "data"]}}]},
            {"label": "DATASET", "pattern": [{"LOWER": {"IN": ["imagenet", "coco", "mnist", "cifar", "glue", "squad", "wikitext", "common", "crawl"]}}]},
            {"label": "DATASET", "pattern": [{"TEXT": {"REGEX": r"[A-Z]{2,}[-]?[0-9]*"}}]},  # CIFAR-10, BERT-Base, etc.
            
            # Method patterns (broad coverage)
            {"label": "METHOD", "pattern": [{"TEXT": {"REGEX": r"[A-Z]{2,8}"}}, {"LOWER": {"IN": ["model", "network", "algorithm"]}}]},  # CNN model, BERT network
            {"label": "METHOD", "pattern": [{"LOWER": {"IN": ["transformer", "bert", "gpt", "resnet", "vgg", "alexnet", "lstm", "gru", "rnn"]}}]},
            {"label": "METHOD", "pattern": [{"LOWER": {"IN": ["neural", "deep", "machine"]}}, {"LOWER": {"IN": ["network", "learning", "model"]}}]},
            {"label": "METHOD", "pattern": [{"LOWER": {"IN": ["random", "decision", "gradient"]}}, {"LOWER": {"IN": ["forest", "tree", "boosting", "descent"]}}]},
            {"label": "METHOD", "pattern": [{"LOWER": {"IN": ["logistic", "linear", "polynomial"]}}, {"LOWER": "regression"}]},
            {"label": "METHOD", "pattern": [{"LOWER": {"IN": ["support", "k-nearest", "naive"]}}, {"LOWER": {"IN": ["vector", "neighbor", "bayes"]}}]},
            {"label": "METHOD", "pattern": [{"LOWER": {"IN": ["reinforcement", "supervised", "unsupervised", "semi-supervised"]}}, {"LOWER": "learning"}]},
            
            # Organization patterns (universities, companies, institutions)
            {"label": "ORG", "pattern": [{"TEXT": {"REGEX": r"[A-Z][a-zA-Z]+"}}, {"LOWER": {"IN": ["university", "institute", "institution", "college", "school"]}}]},
            {"label": "ORG", "pattern": [{"TEXT": {"REGEX": r"[A-Z][a-zA-Z]+"}}, {"LOWER": {"IN": ["inc", "corp", "llc", "ltd", "co", "company"]}}]},
            {"label": "ORG", "pattern": [{"TEXT": {"REGEX": r"[A-Z][a-zA-Z]+"}}, {"LOWER": {"IN": ["lab", "labs", "laboratory", "research", "center", "centre"]}}]},
            {"label": "ORG", "pattern": [{"LOWER": {"IN": ["google", "microsoft", "facebook", "meta", "amazon", "apple", "nvidia", "intel", "ibm", "openai", "anthropic"]}}]},
            {"label": "ORG", "pattern": [{"LOWER": {"IN": ["stanford", "mit", "harvard", "berkeley", "cmu", "oxford", "cambridge", "caltech"]}}]},
            
            # Metric patterns (evaluation measures)
            {"label": "METRIC", "pattern": [{"LOWER": {"IN": ["accuracy", "precision", "recall", "f1", "bleu", "rouge", "meteor", "cider"]}}, {"LOWER": {"IN": ["score", "measure", "metric", ""]}}]},
            {"label": "METRIC", "pattern": [{"LOWER": {"IN": ["mean", "root", "average"]}}, {"LOWER": {"IN": ["squared", "absolute", "pooling"]}}, {"LOWER": "error"}]},
            {"label": "METRIC", "pattern": [{"TEXT": {"REGEX": r"\w+"}}, {"TEXT": "-"}, {"LOWER": {"IN": ["score", "measure", "metric", "ratio", "rate"]}}]},
            {"label": "METRIC", "pattern": [{"LOWER": {"IN": ["auc", "roc", "map", "ndcg", "mrr", "top", "hit"]}}, {"TEXT": {"REGEX": r"[@-]?\d*"}}]},
            
            # Temporal patterns
            {"label": "TEMPORAL", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years", "month", "months", "day", "days", "hour", "hours"]}}]},
            {"label": "TEMPORAL", "pattern": [{"LOWER": {"IN": ["daily", "weekly", "monthly", "quarterly", "annually", "yearly"]}}]},
            {"label": "TEMPORAL", "pattern": [{"LOWER": {"IN": ["pre", "post", "next", "previous"]}}, {"TEXT": "-"}, {"LOWER": {"IN": ["training", "processing", "evaluation"]}}]},
            
            # Architecture/Framework patterns
            {"label": "FRAMEWORK", "pattern": [{"LOWER": {"IN": ["pytorch", "tensorflow", "keras", "scikit", "huggingface", "transformers"]}}, {"LOWER": {"IN": ["learn", "face", ""]}}]},
            {"label": "FRAMEWORK", "pattern": [{"LOWER": {"IN": ["pipeline", "architecture", "framework", "system", "platform", "infrastructure"]}}]},
            
            # Process/Mechanism patterns
            {"label": "PROCESS", "pattern": [{"LOWER": {"IN": ["training", "inference", "evaluation", "testing", "validation", "preprocessing", "postprocessing"]}}]},
            {"label": "PROCESS", "pattern": [{"LOWER": {"IN": ["encoding", "decoding", "embedding", "tokenization", "normalization", "optimization"]}}]},
        ]
        
        ruler.add_patterns(patterns)

    def setup_ml_models(self):
        """Initialize ML-based NER models"""
        try:
            print("Loading SciBERT for scientific entity recognition...")
            self.scibert_ner = pipeline(
                "ner",
                model="allenai/scibert_scivocab_uncased",
                tokenizer="allenai/scibert_scivocab_uncased",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Fallback general NER
            self.general_ner = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("✅ ML models loaded successfully")
            
        except Exception as e:
            print(f"⚠️ ML models failed to load: {e}")
            print("Continuing with spaCy + patterns only")
            self.scibert_ner = None
            self.general_ner = None

    def extract_ml_entities(self, text: str) -> List[Dict]:
        """Extract entities using ML models"""
        all_entities = []
        
        # SciBERT for scientific entities
        if self.scibert_ner:
            try:
                scibert_entities = self.scibert_ner(text)
                for entity in scibert_entities:
                    all_entities.append({
                        "text": entity["word"],
                        "start": entity["start"],
                        "end": entity["end"],
                        "label": entity["entity_group"],
                        "confidence": entity["score"],
                        "source": "scibert"
                    })
            except Exception as e:
                print(f"SciBERT error: {e}")
        
        # General NER as fallback
        if self.general_ner:
            try:
                general_entities = self.general_ner(text)
                for entity in general_entities:
                    # Avoid duplicates
                    is_duplicate = any(
                        abs(entity["start"] - existing["start"]) < 5 
                        for existing in all_entities
                    )
                    if not is_duplicate:
                        all_entities.append({
                            "text": entity["word"],
                            "start": entity["start"],
                            "end": entity["end"],
                            "label": entity["entity_group"],
                            "confidence": entity["score"],
                            "source": "general"
                        })
            except Exception as e:
                print(f"General NER error: {e}")
        
        return all_entities

    def extract_pattern_entities(self, text: str) -> List[Dict]:
        """Extract entities using comprehensive pattern matching"""
        entities = []
        
        # Academic method patterns
        method_patterns = [
            r'\b(?:neural network|deep learning|machine learning|CNN|LSTM|GRU|RNN)\b',
            r'\b(?:transformer|BERT|GPT|ResNet|VGG|AlexNet)\b',
            r'\b(?:SVM|random forest|decision tree|logistic regression)\b',
            r'\b(?:clustering|classification|regression|reinforcement learning)\b',
            r'\b[A-Z]{2,6}(?:-[A-Z]{2,6})*\b',  # Acronyms
        ]
        
        # Dataset patterns
        dataset_patterns = [
            r'\b(?:ImageNet|COCO|MNIST|CIFAR|WikiText|Common Crawl)\b',
            r'\b[A-Z][a-zA-Z0-9]*(?:\s+)?(?:dataset|corpus|database)\b',
            r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Dataset|Corpus|DB)\b'
        ]
        
        # Temporal patterns
        temporal_patterns = [
            r'\b\d{4}\s*(?:to|through|-|–)\s*\d{4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(?:daily|weekly|monthly|annually|quarterly)\b',
            r'\b(?:next-\w+|pre-\w+|post-\w+)\b'
        ]
        
        # Organization patterns
        org_patterns = [
            r'\b(?:Google|Microsoft|Facebook|Meta|Amazon|Apple|NVIDIA|Intel|IBM)\b',
            r'\b(?:Stanford|MIT|Harvard|Berkeley|CMU|Oxford|Cambridge)\b',
            r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Inc|Corp|LLC|Ltd|University|Institute|Lab|Labs)\b'
        ]
        
        # Mechanism patterns
        mechanism_patterns = [
            r'\b(?:pipeline|architecture|framework|system)\b',
            r'\b(?:stages?|components?|modules?)\b',
            r'\b(?:inference|processing|classification)\b'
        ]
        
        # Relational patterns
        relational_patterns = [
            r'\b(?:three|four|five)\s+(?:main\s+)?(?:stages?|components?|steps?)\b',
            r'\b(?:first|second|third|final)\s+(?:stage|step|phase)\b',
            r'\b(?:consists? of|composed of|divided into)\b'
        ]
        
        # Causal patterns
        causal_patterns = [
            r'\b(?:outputs?|results? in|leads to|causes?)\b',
            r'\b(?:fed into|processed by|transforms?)\b',
            r'\b(?:probability of|predicted|classification)\b'
        ]
        
        # Apply all patterns
        pattern_groups = [
            (method_patterns, "METHOD"),
            (dataset_patterns, "DATASET"), 
            (temporal_patterns, "TEMPORAL"),
            (org_patterns, "ORG"),
            (mechanism_patterns, "MECHANISM"),
            (relational_patterns, "RELATIONAL"),
            (causal_patterns, "CAUSAL")
        ]
        
        for patterns, label in pattern_groups:
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append({
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "label": label,
                        "confidence": 0.7,
                        "source": "pattern"
                    })
        
        return entities

    def identify_maskable_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Main entity identification using ML + patterns + spaCy"""
        
        # Extract using all methods
        ml_entities = self.extract_ml_entities(text) if (self.scibert_ner or self.general_ner) else []
        pattern_entities = self.extract_pattern_entities(text)
        
        # spaCy entities
        doc = self.nlp(text)
        spacy_entities = []
        for ent in doc.ents:
            spacy_entities.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
                "confidence": 0.8,
                "source": "spacy"
            })
        
        # Merge and deduplicate
        all_entities = self.merge_and_deduplicate([ml_entities, pattern_entities, spacy_entities])
        
        # Categorize for reasoning types
        categorized = self.categorize_entities(all_entities)
        
        return categorized

    def merge_and_deduplicate(self, entity_lists: List[List[Dict]]) -> List[Dict]:
        """Merge entities from different sources and remove duplicates"""
        all_entities = []
        for entity_list in entity_lists:
            all_entities.extend(entity_list)
        
        # Sort by confidence
        all_entities.sort(key=lambda x: x.get("confidence", 0.5), reverse=True)
        
        # Remove duplicates
        unique_entities = []
        for entity in all_entities:
            is_duplicate = False
            for existing in unique_entities:
                # Check overlap
                overlap_start = max(entity["start"], existing["start"])
                overlap_end = min(entity["end"], existing["end"])
                overlap_length = max(0, overlap_end - overlap_start)
                
                entity_length = entity["end"] - entity["start"]
                overlap_ratio = overlap_length / entity_length if entity_length > 0 else 0
                
                if overlap_ratio > 0.5:  # 50% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_entities.append(entity)
        
        return unique_entities

    def categorize_entities(self, entities: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize entities into reasoning type groups"""
        categorized = {
            "methods": [],
            "mechanisms": [],
            "temporal_markers": [],
            "professional_entities": [],
            "relational_entities": [],
            "causal_entities": []
        }
        
        for entity in entities:
            label = entity["label"].upper()
            text = entity["text"].lower()
            
            # Categorization logic
            if label in ["METHOD", "ALGORITHM", "MODEL", "TECHNIQUE"] or any(
                keyword in text for keyword in ["network", "model", "algorithm", "learning", "regression"]
            ):
                categorized["methods"].append(entity)
                
            elif label in ["ORG", "ORGANIZATION", "DATASET", "PERSON"] or any(
                keyword in text for keyword in ["university", "institute", "corp", "inc", "lab", "dataset"]
            ):
                categorized["professional_entities"].append(entity)
                
            elif label in ["DATE", "TIME", "TEMPORAL"] or any(
                keyword in text for keyword in ["month", "year", "daily", "weekly", "annual"]
            ) or re.search(r'\d{4}', text):
                categorized["temporal_markers"].append(entity)
                
            elif label in ["MECHANISM"] or any(
                keyword in text for keyword in ["pipeline", "architecture", "framework", "system", "stage", "component"]
            ):
                categorized["mechanisms"].append(entity)
                
            elif label in ["CAUSAL"] or any(
                keyword in text for keyword in ["output", "result", "produce", "generate", "predict", "fed into"]
            ):
                categorized["causal_entities"].append(entity)
                
            elif label in ["RELATIONAL"] or any(
                keyword in text for keyword in ["consist", "compose", "three", "stages", "components"]
            ):
                categorized["relational_entities"].append(entity)
        
        return categorized

    def calculate_masking_score(self, text: str, entity: Dict, reasoning_type: str) -> float:
        """Calculate score for masking an entity"""
        base_score = len(entity["text"])
        
        # Confidence bonus
        confidence_bonus = entity.get("confidence", 0.5) * 10
        
        # Source bonus
        source_bonus = {
            "scibert": 15,
            "general": 10,
            "spacy": 5,
            "pattern": 3
        }.get(entity.get("source", "pattern"), 0)
        
        # Reasoning-specific bonuses
        reasoning_bonus = 0
        if reasoning_type == "professional_entity_location":
            if entity.get("label") in ["ORG", "DATASET", "INSTITUTION"]:
                reasoning_bonus = 10
        elif reasoning_type == "methodological_technical_reasoning":
            if entity.get("label") in ["METHOD", "ALGORITHM", "MODEL"]:
                reasoning_bonus = 12
        elif reasoning_type == "multimodal_temporal_reasoning":
            if entity.get("label") in ["TEMPORAL", "DATE", "TIME"]:
                reasoning_bonus = 15
        
        total_score = base_score + confidence_bonus + source_bonus + reasoning_bonus
        
        # Penalty for very short entities
        if len(entity["text"]) < 3:
            total_score -= 10
        
        return total_score

    def create_contextual_gaps(self, text: str, entities: Dict, reasoning_type: str) -> Dict:
        """Create contextual gaps for a specific reasoning type"""
        config = self.reasoning_configs[reasoning_type]
        
        # Select candidate entities
        if reasoning_type == "professional_entity_location":
            candidates = entities["professional_entities"]
        elif reasoning_type == "multimodal_temporal_reasoning":
            candidates = entities["temporal_markers"]
        elif reasoning_type == "cross_subgraph_role_reasoning":
            candidates = entities["relational_entities"] + entities["mechanisms"]
        elif reasoning_type == "causal_mechanism_reasoning":
            candidates = entities["causal_entities"] + entities["mechanisms"]
        elif reasoning_type == "methodological_technical_reasoning":
            candidates = entities["methods"]
        else:
            candidates = []
        
        if not candidates:
            return None
        
        # Score candidates
        scored_candidates = []
        for entity in candidates:
            score = self.calculate_masking_score(text, entity, reasoning_type)
            if score > config["threshold"]:
                scored_candidates.append((entity, score))
        
        if not scored_candidates:
            return None
        
        # Sort and select
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        num_masks = min(len(scored_candidates), config["max_masks"])
        num_masks = max(num_masks, config["min_masks"]) if len(scored_candidates) >= config["min_masks"] else len(scored_candidates)
        
        selected_entities = [item[0] for item in scored_candidates[:num_masks]]
        
        # Create gapped text
        gapped_text = self.apply_masks(text, selected_entities)
        
        return {
            "reasoning_type": reasoning_type,
            "competency": self.get_competency_description(reasoning_type),
            "original_text": text,
            "gapped_text": gapped_text,
            "masked_entities": selected_entities,
            "answers": [entity["text"] for entity in sorted(selected_entities, key=lambda x: x["start"])],  # Sort answers by text position
            "confidence_scores": [float(entity.get("confidence", 0.5)) for entity in sorted(selected_entities, key=lambda x: x["start"])]  # Sort confidence scores too
        }

    def apply_masks(self, text: str, entities: List[Dict]) -> str:
        """Apply numbered masks to create fill-in-the-blank text"""
        # Sort entities by start position (ascending for logical order)
        entities_sorted_for_numbering = sorted(entities, key=lambda x: x["start"])
        
        # Create position-to-number mapping
        position_to_number = {}
        for i, entity in enumerate(entities_sorted_for_numbering):
            position_to_number[entity["start"]] = i + 1
        
        # Sort entities by start position (reverse order for correct replacement)
        entities_sorted_for_replacement = sorted(entities, key=lambda x: x["start"], reverse=True)
        
        masked_text = text
        for entity in entities_sorted_for_replacement:
            # Use the logical number based on text order
            number = position_to_number[entity["start"]]
            blank = f"______{number}______"
            start, end = entity["start"], entity["end"]
            masked_text = masked_text[:start] + blank + masked_text[end:]
        
        return masked_text

    def get_competency_description(self, reasoning_type: str) -> str:
        """Get competency description for each reasoning type"""
        competencies = {
            "professional_entity_location": "domain knowledge recognition",
            "multimodal_temporal_reasoning": "time-dependent process understanding across modalities",
            "cross_subgraph_role_reasoning": "inter-component relationship analysis",
            "causal_mechanism_reasoning": "cause-effect understanding",
            "methodological_technical_reasoning": "research method comprehension"
        }
        return competencies.get(reasoning_type, "general reasoning")

    def analyze_discourse_structure(self, text: str, entities: Dict) -> List[str]:
        """Determine applicable reasoning types"""
        applicable_types = []
        
        if entities["professional_entities"]:
            applicable_types.append("professional_entity_location")
        
        if entities["temporal_markers"]:
            applicable_types.append("multimodal_temporal_reasoning")
        
        if entities["relational_entities"] or entities["mechanisms"]:
            applicable_types.append("cross_subgraph_role_reasoning")
        
        if entities["causal_entities"]:
            applicable_types.append("causal_mechanism_reasoning")
        
        if entities["methods"]:
            applicable_types.append("methodological_technical_reasoning")
        
        return applicable_types

    def process_dataset_entry(self, entry: Dict) -> List[Dict]:
        """Process a single dataset entry to generate questions"""
        # Extract text
        text = entry["dual_analysis"]["vision_enhanced_analysis"]
        
        # Identify entities
        entities = self.identify_maskable_entities(text)
        
        # Determine applicable reasoning types
        applicable_types = self.analyze_discourse_structure(text, entities)
        
        # Generate questions
        questions = []
        for reasoning_type in applicable_types:
            gap_result = self.create_contextual_gaps(text, entities, reasoning_type)
            if gap_result:
                gap_result.update({
                    "paper_id": entry["paper_id"],
                    "figure_id": entry["figure_id"],
                    "figure_path": entry.get("figure_path", ""),
                    "caption": entry.get("caption", "")
                })
                questions.append(gap_result)
        
        return questions

def run_pipeline(input_path: str, output_path: str):
    """Process entire dataset - handles both JSON and JSONL formats"""
    gap_creator = ContextualGapCreator()
    
    # Check if input is JSONL or JSON
    if input_path.endswith('.jsonl'):
        # Read JSONL format
        records = []
        with open(input_path, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Error parsing line {line_num}: {e}")
                        continue
    else:
        # Read JSON format
        with open(input_path, "r", encoding='utf-8') as f:
            records = json.load(f)
    
    print(f"📊 Processing {len(records)} records from {input_path}")
    
    all_questions = []
    successful_records = 0
    failed_records = 0
    
    for i, record in enumerate(records):
        try:
            paper_id = record.get('paper_id', f'record_{i+1}')
            print(f"Processing record {i+1}/{len(records)}: {paper_id}")
            
            # Check if record has required fields
            if 'dual_analysis' not in record:
                print(f"  ⚠️ Skipping - no 'dual_analysis' field")
                failed_records += 1
                continue
                
            if 'vision_enhanced_analysis' not in record['dual_analysis']:
                print(f"  ⚠️ Skipping - no 'vision_enhanced_analysis' field")
                failed_records += 1
                continue
            
            # Process the record
            questions = gap_creator.process_dataset_entry(record)
            all_questions.extend(questions)
            successful_records += 1
            
            print(f"  ✅ Generated {len(questions)} questions")
            
        except Exception as e:
            print(f"  ❌ Error processing record {i+1}: {e}")
            failed_records += 1
            continue
    
    # Save as JSON (array of objects)
    with open(output_path, "w", encoding='utf-8') as f:
        # Convert all questions to serializable format
        serializable_questions = [convert_to_serializable(question) for question in all_questions]
        json.dump(serializable_questions, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 PIPELINE COMPLETED")
    print(f"✅ Generated {len(all_questions)} questions saved to {output_path}")
    print(f"📈 Success rate: {successful_records}/{len(records)} records ({successful_records/len(records)*100:.1f}%)")
    
    if failed_records > 0:
        print(f"⚠️ Failed records: {failed_records}")
    
    # Statistics by reasoning type
    by_type = {}
    for q in all_questions:
        rtype = q["reasoning_type"]
        by_type[rtype] = by_type.get(rtype, 0) + 1
    
    print(f"\n📊 Questions by reasoning type:")
    for rtype, count in sorted(by_type.items()):
        print(f"  {rtype}: {count}")
    
    return len(all_questions), successful_records, failed_records

def convert_to_serializable(obj):
    """Convert numpy/torch types to JSON serializable types"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle torch tensors if present
    elif hasattr(obj, 'detach') and hasattr(obj, 'cpu'):  # torch tensor
        return obj.detach().cpu().numpy().tolist()
    # Handle numpy float32/float64
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj

# Example usage
if __name__ == "__main__":
    # Process finance dataset directly
    print("🚀 Starting Contextual Gap Creation Pipeline")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs("output", exist_ok=True)
    
    # Run the pipeline on your dataset
    total_questions, successful_records, failed_records = run_pipeline(
        "dataset/finance_dual_distilled.jsonl", 
        "output/stage2_gap_questions2.json"  # Changed to .json
    )
    
    print(f"\n🎯 PIPELINE SUMMARY:")
    print(f"   Total Questions Generated: {total_questions}")
    print(f"   Records Processed Successfully: {successful_records}")
    print(f"   Records Failed: {failed_records}")
    print(f"   Output File: output/stage2_gap_questions1.json")
    
    # Optional: Test with single entry for debugging
    if total_questions == 0:
        print("\n🔍 No questions generated. Testing with single entry...")
        gap_creator = ContextualGapCreator()
        
        # Test dataset entry
        test_entry = {
            "paper_id": "test-entry",
            "figure_id": "fig1", 
            "figure_path": "test.png",
            "caption": "Test Figure",
            "dual_analysis": {
                "vision_enhanced_analysis": "The study utilizes machine learning algorithms including Random Forest and LSTM models to predict financial outcomes using data from 2020 to 2023."
            }
        }
        
        questions = gap_creator.process_dataset_entry(test_entry)
        print(f"✅ Test entry generated {len(questions)} questions")
        
        for i, question in enumerate(questions, 1):
            print(f"\nTest Question {i}: {question['reasoning_type']}")
            print(f"Question: {question['gapped_text']}")
            print(f"Answers: {question['answers']}")
    
    print("\n✨ Pipeline execution completed!")
    print("="*60)