import os
import re
import json
import base64
import io
import time
from datetime import datetime
from collections import Counter

# Dependencies
import fitz  # PyMuPDF
import google.generativeai as genai
import nltk
from textblob import TextBlob
from flask import Flask, request, jsonify, session
from flask_session import Session
from pymongo import MongoClient
from bson.objectid import ObjectId
from bson.binary import Binary
from google.api_core.exceptions import ResourceExhausted

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'supersecretkey')
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# MongoDB Connection
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client["test_platform"]
testdb = client['test']
questions_collection = db["questions"]
students_collection = db["students"]
results_collection = db["results"]
tests_collection = testdb["tests"]
feedback_collection = db["feedback"]
resources_collection = db["resources"]

# Get API key from environment
GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY')

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-latest")

def extract_text_from_pdf_base64(base64_string):
    """Extract text from a base64-encoded PDF"""
    try:
        base64_string = base64_string.strip()
        
        # Decode Base64 string to bytes
        pdf_bytes = base64.b64decode(base64_string)
        
        # Ensure the file starts with '%PDF'
        if not pdf_bytes.startswith(b"%PDF"):
            return "❌ Error: Decoded data is not a valid PDF file."
        
        # Create a file-like object from bytes
        pdf_stream = io.BytesIO(pdf_bytes)
        
        # Open the PDF from memory
        doc = fitz.open("pdf", pdf_stream)
        
        # Extract text
        text = "\n".join(page.get_text("text") for page in doc)
        
        return text if text else "❌ Error: No extractable text found in the PDF."
    
    except Exception as e:
        return f"❌ Error reading PDF: {str(e)}"

def extract_questions(text):
    """Extract questions and marks from text"""
    # Match patterns like "1. Question text (5 marks)" or "2. Question text (2 marks)"
    pattern = r"(\d+\.\s*[^\(]+)\s*\((\d+)\s*[Mm]arks?\)"
    matches = re.findall(pattern, text)
    return [{"question": q.strip(), "marks": int(m)} for q, m in matches]

def extract_important_keywords(text):
    """Extract important keywords from text using NLTK"""
    # Remove stop words and get meaningful words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [word.lower() for word in nltk.word_tokenize(text) 
             if word.isalnum() and word.lower() not in stop_words and len(word) > 3]
    
    # Use FreqDist to find important words
    freq_dist = nltk.FreqDist(words)
    keywords = [word for word, freq in freq_dist.most_common(10)]
    
    return keywords

def check_grammar(text):
    """Check grammar issues in student answers"""
    if not text or len(text.strip()) < 10:
        return []
    
    issues = []
    
    try:
        # Use TextBlob for basic grammar checking
        blob = TextBlob(text)
        
        # Check for simple grammar issues
        sentences = blob.sentences
        for sentence in sentences:
            # Check capitalization of first letter
            if len(str(sentence)) > 0 and str(sentence)[0].islower():
                issues.append(f"Sentence should start with a capital letter: '{str(sentence)[:30]}...'")
            
            # Check for ending punctuation
            if not str(sentence).endswith(('.', '!', '?')):
                issues.append(f"Sentence missing ending punctuation: '{str(sentence)[:30]}...'")
        
        # Check for repeated words
        words = text.lower().split()
        for i in range(len(words)-1):
            if words[i] == words[i+1] and words[i] not in ['the', 'a', 'an']:
                issues.append(f"Repeated word: '{words[i]}'")
        
        return issues[:5]  # Limit to top 5 issues
    except Exception as e:
        print(f"Grammar check error: {str(e)}")
        return []

def get_resources_for_concepts(concepts):
    """Find relevant educational resources for concepts"""
    resources = []
    
    # First try to find exact matches
    for concept in concepts:
        concept_resources = list(resources_collection.find(
            {'topics': {'$regex': concept, '$options': 'i'}}
        ).limit(2))
        
        resources.extend(concept_resources)
    
    # If we didn't find enough resources, get general recommendations
    if len(resources) < 3:
        general_resources = list(resources_collection.find().limit(3 - len(resources)))
        resources.extend(general_resources)
    
    # Format resource data
    formatted_resources = []
    for resource in resources:
        formatted_resources.append({
            'title': resource.get('title', 'Untitled Resource'),
            'type': resource.get('type', 'Unknown'),
            'url': resource.get('url', '#'),
            'description': resource.get('description', 'No description available')
        })
    
    return formatted_resources

def enhanced_grading(question, answer, max_marks):
    """Fallback grading method when Gemini API fails"""
    # Simple keyword matching for fallback grading
    question_keywords = extract_important_keywords(question)
    answer_keywords = extract_important_keywords(answer)
    
    # Find common keywords
    common_keywords = set(answer_keywords) & set(question_keywords)
    
    # Calculate a basic score based on keyword matching
    match_percentage = len(common_keywords) / max(len(question_keywords), 1)
    score = int(match_percentage * max_marks)
    
    # Ensure score doesn't exceed max_marks
    score = min(score, max_marks)
    
    # Missing keywords
    missing_keywords = list(set(question_keywords) - set(answer_keywords))
    
    # Generate feedback
    if score >= 0.8 * max_marks:
        feedback = "Good answer with relevant keywords."
    elif score >= 0.5 * max_marks:
        feedback = "Partial answer with some key concepts missing."
    else:
        feedback = "Answer lacks many relevant keywords and concepts."
    
    return {
        "marks": score,
        "feedback": feedback,
        "weak_concepts": ["Content coverage", "Key terminology"],
        "improvement_suggestions": [
            "Include more specific terminology", 
            "Expand on core concepts"
        ],
        "resource_topics": [question.split()[0]],
        "grammar_issues": check_grammar(answer),
        "missing_keywords": missing_keywords[:5]
    }

def grade_answer_with_gemini(question, answer, max_marks):
    """Grade answers using Gemini API with error handling and fallback"""
    # Check for empty answer first
    if not answer or answer.strip() == '':
        return {
            "marks": 0, 
            "feedback": "No answer provided.",
            "weak_concepts": ["Content knowledge"],
            "improvement_suggestions": ["Please attempt to answer the question"],
            "grammar_issues": [],
            "missing_keywords": []
        }
    
    # Try using Gemini API
    try:
        # Construct prompt for Gemini
        prompt = f"""
        Question: {question}
        
        Student Answer: {answer}
        
        Please evaluate this answer out of {max_marks} marks and provide:
        1. A score out of {max_marks} (as an integer)
        2. Brief constructive feedback (2-3 sentences)
        3. List of weak concepts (up to 3)
        4. List of improvement suggestions (up to 3)
        5. List of relevant topics for further study (up to 3)
        
        Format your response as JSON:
        {{
          "marks": <score>,
          "feedback": "<feedback>",
          "weak_concepts": ["<concept1>", "<concept2>", ...],
          "improvement_suggestions": ["<suggestion1>", "<suggestion2>", ...],
          "resource_topics": ["<topic1>", "<topic2>", ...]
        }}
        
        Ensure your evaluation is fair and aligned with educational standards for this level of question.
        """
        
        # Make API call with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                
                # Extract JSON from response
                response_text = response.text
                
                # Try to find JSON pattern in response
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text
                
                # Clean up any non-JSON parts
                json_str = re.sub(r'^[^{]*', '', json_str)
                json_str = re.sub(r'[^}]*$', '', json_str)
                
                # Parse JSON
                try:
                    result = json.loads(json_str)
                except:
                    # Using eval as a fallback if json.loads fails
                    # Note: eval has security implications in production
                    result = eval(json_str)  
                
                # Ensure all expected fields are present
                if "marks" not in result:
                    result["marks"] = int(max_marks / 2)  # Default to 50%
                else:
                    # Ensure marks don't exceed max_marks
                    result["marks"] = min(int(result["marks"]), max_marks)
                
                if "feedback" not in result or not result["feedback"]:
                    result["feedback"] = "Answer demonstrates partial understanding of the topic."
                
                if "weak_concepts" not in result or not result["weak_concepts"]:
                    result["weak_concepts"] = ["Content knowledge"]
                
                if "improvement_suggestions" not in result or not result["improvement_suggestions"]:
                    result["improvement_suggestions"] = ["Review core concepts related to this topic."]
                
                if "resource_topics" not in result or not result["resource_topics"]:
                    result["resource_topics"] = ["General " + question.split()[0]]
                
                # Get grammar issues
                result["grammar_issues"] = check_grammar(answer)
                
                return result
                
            except ResourceExhausted:
                # API quota exceeded, wait and retry
                if attempt < max_retries - 1:
                    print(f"API quota exceeded, retrying in {(attempt + 1) * 2} seconds...")
                    time.sleep((attempt + 1) * 2)
                else:
                    raise
            except Exception as e:
                print(f"Gemini API error: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise
    
    except Exception as e:
        print(f"Gemini evaluation failed: {str(e)}")
        # Fall back to the enhanced_grading method
        return enhanced_grading(question, answer, max_marks)
    
import time
import os
import random
from functools import wraps
import json
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError

# Load API key from environment
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
MAX_RETRIES = int(os.environ.get('GEMINI_MAX_RETRIES', 3))
RETRY_DELAY = int(os.environ.get('GEMINI_RETRY_DELAY', 5))

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Implement a rate-limiting decorator
def with_rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (ResourceExhausted, ServiceUnavailable, InternalServerError) as e:
                if attempt < MAX_RETRIES - 1:
                    # Add jitter to prevent thundering herd
                    sleep_time = RETRY_DELAY * (attempt + 1) + random.uniform(0, 2)
                    print(f"API limit reached, retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"Max retries reached: {str(e)}")
                    raise
    return wrapper

# Implement a request batcher to process multiple questions in batches
class GeminiBatcher:
    def __init__(self, batch_size=5, delay_between_batches=10):
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
    
    def process_items(self, items, process_func):
        """Process items in batches with delay between batches"""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Process this batch
            batch_results = [process_func(item) for item in batch]
            results.extend(batch_results)
            
            # Wait between batches if there are more to process
            if i + self.batch_size < len(items):
                print(f"Processed batch {i//self.batch_size + 1}, waiting {self.delay_between_batches} seconds...")
                time.sleep(self.delay_between_batches)
        
        return results

@with_rate_limit
def chat_with_gemini(prompt):
    """Send a request to Gemini API with retry logic and error handling"""
    try:
        response = model.generate_content(prompt)
        
        # Extract JSON response
        response_text = response.text
        
        # Try to find JSON pattern in response
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text
        
        # Clean up any non-JSON parts
        json_str = json_str.strip()
        json_str = re.sub(r'^[^{]*', '', json_str)
        json_str = re.sub(r'[^}]*$', '', json_str)
        
        # Parse JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Using eval as a fallback if json.loads fails
            return eval(json_str)
    except Exception as e:
        print(f"Error in Gemini API call: {str(e)}")
        return {"error": str(e)}

def grade_answer_with_gemini(question, answer, max_marks):
    """Grade a single answer using Gemini"""
    if not answer or answer.strip() == '':
        return {
            "marks": 0, 
            "feedback": "No answer provided.",
            "weak_concepts": ["Content knowledge"],
            "improvement_suggestions": ["Please attempt to answer the question"],
            "grammar_issues": [],
            "missing_keywords": []
        }
    
    try:
        prompt = f"""
        Question: {question}
        
        Student Answer: {answer}
        
        Please evaluate this answer out of {max_marks} marks and provide:
        1. A score out of {max_marks} (as an integer)
        2. Brief constructive feedback (2-3 sentences)
        3. List of weak concepts (up to 3)
        4. List of improvement suggestions (up to 3)
        5. List of relevant topics for further study (up to 3)
        
        Format your response as JSON:
        {{
          "marks": <score>,
          "feedback": "<feedback>",
          "weak_concepts": ["<concept1>", "<concept2>", ...],
          "improvement_suggestions": ["<suggestion1>", "<suggestion2>", ...],
          "resource_topics": ["<topic1>", "<topic2>", ...]
        }}
        """
        
        result = chat_with_gemini(prompt)
        
        # Ensure all expected fields are present
        if "marks" not in result:
            result["marks"] = int(max_marks / 2)  # Default to 50%
        else:
            # Ensure marks don't exceed max_marks
            result["marks"] = min(int(result["marks"]), max_marks)
        
        if "feedback" not in result or not result["feedback"]:
            result["feedback"] = "Answer demonstrates partial understanding of the topic."
        
        if "weak_concepts" not in result or not result["weak_concepts"]:
            result["weak_concepts"] = ["Content knowledge"]
        
        if "improvement_suggestions" not in result or not result["improvement_suggestions"]:
            result["improvement_suggestions"] = ["Review core concepts related to this topic."]
        
        if "resource_topics" not in result or not result["resource_topics"]:
            result["resource_topics"] = ["General " + question.split()[0]]
        
        return result
    except Exception as e:
        print(f"Grading failed: {str(e)}")
        # Fall back to the enhanced_grading method
        return enhanced_grading(question, answer, max_marks)

@app.route('/process_questions/<test_id>', methods=['GET'])
def process_questions(test_id):
    """Process questions from a PDF in a test document"""
    if not test_id:
        return jsonify({"error": "Test ID is required"}), 400
    
    try:
        # Retrieve the test document from the database
        test = tests_collection.find_one({'testid': ObjectId(test_id)})
        
        if not test or 'file' not in test:
            return jsonify({"error": "Test or associated PDF file not found"}), 404
        
        pdf_data = test['file']
        
        # Convert BSON Binary to Base64 if necessary
        if isinstance(pdf_data, Binary):  # MongoDB stores it as Binary
            pdf_data = base64.b64encode(pdf_data).decode('utf-8')
        elif isinstance(pdf_data, bytes):  # Raw binary data
            pdf_data = base64.b64encode(pdf_data).decode('utf-8')
        
        extracted_text = extract_text_from_pdf_base64(pdf_data)
        
        if "Error" in extracted_text:
            return jsonify({"error": extracted_text}), 500
        
        questions = extract_questions(extracted_text)
        
        if not questions:
            return jsonify({"error": "No questions could be extracted from the PDF"}), 400
        
        # Sort extracted questions into categories
        sorted_questions = {
            "MCQ": [q["question"] for q in questions if q["marks"] == 1],  # If MCQs have marks = 1
            "2-Mark": [q["question"] for q in questions if q["marks"] == 2],
            "5-Mark": [q["question"] for q in questions if q["marks"] == 5]
        }
        
        # Update MongoDB
        update_data = {
            "ExtractedQues.MCQ": sorted_questions["MCQ"],
            "ExtractedQues.2-Mark": sorted_questions["2-Mark"],
            "ExtractedQues.5-Mark": sorted_questions["5-Mark"],
            "questions": questions  # Store the full question objects with marks
        }
        
        result = tests_collection.update_one(
            {'testid': ObjectId(test_id)},
            {'$set': update_data}
        )
        
        session['current_test_id'] = test_id
        
        return jsonify({
            "success": True,
            "message": "Questions processed successfully",
            "questions": questions,
            "matched_count": result.matched_count,
            "modified_count": result.modified_count
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_test_questions/<test_id>', methods=['GET'])
def get_test_questions(test_id):
    """Get processed questions for a test"""
    try:
        test = tests_collection.find_one({'testid': ObjectId(test_id)})
        
        if not test:
            return jsonify({"error": "Test not found"}), 404
        
        extracted_questions = test.get('ExtractedQues', {})
        
        # Format the response
        response = {
            "test_id": test_id,
            "test_name": test.get('test_name', 'Unnamed Test'),
            "questions": {
                "MCQ": extracted_questions.get('MCQ', []),
                "2-Mark": extracted_questions.get('2-Mark', []),
                "5-Mark": extracted_questions.get('5-Mark', [])
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/evaluate_answers', methods=['POST'])
def evaluate_answers():
    """Evaluate student answers and provide feedback with rate limiting"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['student_name', 'email', 'test_id', 'answers']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        student_name = data['student_name']
        email = data['email']
        test_id = data['test_id']
        answers = data['answers']
        
        # Retrieve test
        test = tests_collection.find_one({'testid': ObjectId(test_id)})
        if not test:
            return jsonify({"error": "Test not found"}), 404
        
        # Get questions with marks
        test_questions = test.get('questions', [])
        if not test_questions:
            return jsonify({"error": "No questions found for this test"}), 404
        
        # Prepare questions for batch processing
        questions_to_grade = []
        for question in test_questions:
            question_text = question['question']
            max_marks = question['marks']
            student_answer = answers.get(question_text, '')
            
            questions_to_grade.append({
                'question_text': question_text,
                'student_answer': student_answer,
                'max_marks': max_marks
            })
        
        # Process questions in batches
        batcher = GeminiBatcher(batch_size=3, delay_between_batches=5)
        
        def grade_question(item):
            return {
                'question': item['question_text'],
                'grading_result': grade_answer_with_gemini(
                    item['question_text'], 
                    item['student_answer'], 
                    item['max_marks']
                ),
                'student_answer': item['student_answer'],
                'max_marks': item['max_marks']
            }
        
        graded_questions = batcher.process_items(questions_to_grade, grade_question)
        
        # Prepare feedback
        total_score = 0
        max_possible_score = 0
        detailed_feedback = []
        
        for graded in graded_questions:
            result = graded['grading_result']
            
            detailed_feedback.append({
                'question': graded['question'],
                'student_answer': graded['student_answer'],
                'marks': result['marks'],
                'max_marks': graded['max_marks'],
                'feedback': result['feedback'],
                'weak_concepts': result.get('weak_concepts', []),
                'improvement_suggestions': result.get('improvement_suggestions', []),
                'grammar_issues': check_grammar(graded['student_answer']),
                'missing_keywords': result.get('missing_keywords', []),
                'recommended_resources': get_resources_for_concepts(result.get('resource_topics', []))
            })
            
            total_score += result['marks']
            max_possible_score += graded['max_marks']
        
        # Prepare common feedback
        common_weak_concepts = set()
        common_improvement_suggestions = set()
        
        for feedback in detailed_feedback:
            common_weak_concepts.update(feedback.get('weak_concepts', []))
            common_improvement_suggestions.update(feedback.get('improvement_suggestions', []))
        
        # Get general resources
        general_resources = get_resources_for_concepts(list(common_weak_concepts))
        
        # Store result in database
        result_doc = {
            'student_name': student_name,
            'student_id': email,
            'test_id': ObjectId(test_id),
            'date': datetime.utcnow(),
            'total_score': total_score,
            'max_possible_score': max_possible_score,
            'detailed_feedback': detailed_feedback,
            'common_weak_concepts': list(common_weak_concepts),
            'common_improvement_suggestions': list(common_improvement_suggestions),
            'general_resources': general_resources
        }
        
        result_id = results_collection.insert_one(result_doc).inserted_id
        
        # Prepare response
        response = {
            'success': True,
            'result_id': str(result_id),
            'student_name': student_name,
            'email': email,
            'total_score': total_score,
            'max_possible_score': max_possible_score,
            'detailed_feedback': detailed_feedback,
            'common_weak_concepts': list(common_weak_concepts),
            'common_improvement_suggestions': list(common_improvement_suggestions),
            'general_resources': general_resources,
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_result/<result_id>', methods=['GET'])
def get_result(result_id):
    """Get a specific result by ID"""
    try:
        result = results_collection.find_one({'_id': ObjectId(result_id)})
        
        if not result:
            return jsonify({"error": "Result not found"}), 404
        
        # Convert ObjectId to string for JSON serialization
        result['_id'] = str(result['_id'])
        result['test_id'] = str(result['test_id'])
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "message": "Assessment backend is running"}), 200

# For Vercel serverless functions
if __name__ == "__main__":
    app.run(debug=True)