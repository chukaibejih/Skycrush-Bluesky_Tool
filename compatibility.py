import json
import math
import os
import random
from typing import Dict, List, Any, Tuple
import requests
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import numpy as np
from datetime import datetime
import emoji

# nltk.download('punkt')
# nltk.download('stopwords')


class ImprovedCompatibilityAnalyzer:
    def __init__(self):
       
        # Download all required NLTK resources
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('vader_lexicon')
            nltk.download('punkt_tab')  # Added this line
            
            # Initialize NLTK resources
            self.stop_words = set(stopwords.words('english'))
            
            # Test tokenization to verify resources are loaded
            test_text = "This is a test sentence."
            word_tokenize(test_text)
            
        except LookupError as e:
            print(f"Error downloading NLTK resources: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error during NLTK initialization: {str(e)}")
            raise
        
        # Define interest categories for better theme matching
        self.interest_categories = {
            'tech': ['programming', 'coding', 'software', 'tech', 'ai', 'computer', 'digital'],
            'gaming': ['game', 'gaming', 'ps5', 'xbox', 'nintendo', 'playstation', 'cod', 'steam'],
            'creative': ['art', 'music', 'writing', 'creative', 'design', 'draw', 'paint'],
            'academic': ['research', 'study', 'learning', 'science', 'phd', 'academic', 'university'],
            'lifestyle': ['fitness', 'health', 'food', 'travel', 'fashion', 'lifestyle'],
            'social': ['friends', 'family', 'community', 'social', 'party', 'meetup'],
            'professional': ['work', 'business', 'career', 'job', 'professional', 'industry']
        }

        # Get both API keys
        self.GOOGLE_API_KEY_1 = os.getenv("GOOGLE_API_KEY_1")
        self.GOOGLE_API_KEY_2 = os.getenv("GOOGLE_API_KEY_2")
        if not self.GOOGLE_API_KEY_1 or not self.GOOGLE_API_KEY_2:
            raise ValueError("Both GOOGLE_API_KEY_1 and GOOGLE_API_KEY_2 must be set in environment variables")

    def _get_random_api_key(self):
        """Randomly select one of the API keys"""
        return random.choice([self.GOOGLE_API_KEY_1, self.GOOGLE_API_KEY_2])

    def generate_compatibility_analysis(self, 
                                     user1_data: Dict[Any, Any],
                                     user2_data: Dict[Any, Any],
                                     user1_posts: List[Dict],
                                     user2_posts: List[Dict]) -> Dict[str, Any]:
        """Main method to generate comprehensive compatibility analysis"""
        
        # Prepare context with enhanced user data
        context = self._prepare_enhanced_context(user1_data, user2_data, user1_posts, user2_posts)
        
        # Calculate detailed metrics
        metrics = self._calculate_detailed_metrics(context)
        
        # Generate narratives based on the detailed metrics
        narratives = self._generate_enhanced_narratives(context, metrics)
        
        return {
            "user1": user1_data['handle'],
            "user2": user2_data['handle'],
            "user1_name": user1_data['displayName'],
            "user2_name": user2_data['displayName'],
            "user1_avi": user1_data['avatar'],
            "user2_avi": user2_data['avatar'],
            "compatibility_summary": self._generate_summary(metrics),
            "narratives": narratives,
            "metrics": metrics,
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "posts_analyzed": {
                    "user1": len(user1_posts),
                    "user2": len(user2_posts)
                }
            }
        }

    def _prepare_enhanced_context(self, user1_data, user2_data, user1_posts, user2_posts):
        """Prepare enriched context with detailed user analysis"""
        
        def extract_post_text(posts):
            return [post.get('post', {}).get('record', {}).get('text', '') for post in posts]
        
        user1_texts = extract_post_text(user1_posts)
        user2_texts = extract_post_text(user2_posts)
        
        return {
            "user1": {
                "handle": user1_data.get("handle", "Unknown"),
                "display_name": user1_data.get("displayName", "Anonymous"),
                "description": user1_data.get("description", ""),
                "followers_count": user1_data.get("followersCount", 0),
                "posts_count": user1_data.get("postsCount", 0),
                "posts": user1_texts,
                "themes": self._analyze_themes(user1_texts),
                "sentiment_profile": self._analyze_sentiment_profile(user1_texts),
                "interaction_style": self._analyze_interaction_style(user1_texts),
                "interests": self._extract_interests(user1_texts),
                "activity_pattern": self._analyze_activity_pattern(user1_posts)
            },
            "user2": {
                "handle": user2_data.get("handle", "Unknown"),
                "display_name": user2_data.get("displayName", "Anonymous"),
                "description": user2_data.get("description", ""),
                "followers_count": user2_data.get("followersCount", 0),
                "posts_count": user2_data.get("postsCount", 0),
                "posts": user2_texts,
                "themes": self._analyze_themes(user2_texts),
                "sentiment_profile": self._analyze_sentiment_profile(user2_texts),
                "interaction_style": self._analyze_interaction_style(user2_texts),
                "interests": self._extract_interests(user2_texts),
                "activity_pattern": self._analyze_activity_pattern(user2_posts)
            }
        }

    def _analyze_themes(self, posts: List[str]) -> Dict[str, float]:
        """Advanced theme analysis using NLP"""
        if not posts:
            return {}
            
        # Combine all posts and tokenize
        text = ' '.join(posts)
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and single characters
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]
        
        # Calculate word frequencies
        word_freq = Counter(tokens)
        total_words = sum(word_freq.values())
        
        # Map words to categories
        category_scores = {category: 0.0 for category in self.interest_categories}
        for word, count in word_freq.items():
            for category, keywords in self.interest_categories.items():
                if any(keyword in word for keyword in keywords):
                    category_scores[category] += count / total_words
                    
        return category_scores

    def _analyze_sentiment_profile(self, posts: List[str]) -> Dict[str, float]:
        """Create detailed sentiment profile"""
        if not posts:
            return {
                "avg_polarity": 0.0,
                "avg_subjectivity": 0.0,
                "emotional_range": 0.0,
                "sentiment_stability": 0.0
            }
            
        sentiments = [TextBlob(post).sentiment for post in posts]
        polarities = [s.polarity for s in sentiments]
        subjectivities = [s.subjectivity for s in sentiments]
        
        return {
            "avg_polarity": np.mean(polarities),
            "avg_subjectivity": np.mean(subjectivities),
            "emotional_range": np.std(polarities),
            "sentiment_stability": 1 - np.std(subjectivities)
        }

    def _analyze_interaction_style(self, posts: List[str]) -> Dict[str, float]:
        """Analyze user's interaction patterns"""
        if not posts:
            return {
                "avg_length": 0.0,
                "emoji_usage": 0.0,
                "question_frequency": 0.0,
                "response_style": 0.0
            }
            
        total_posts = len(posts)
        total_chars = sum(len(post) for post in posts)
        emoji_count = sum(len([c for c in post if c in emoji.EMOJI_DATA]) for post in posts)
        question_count = sum(post.count('?') for post in posts)
        
        return {
            "avg_length": total_chars / total_posts,
            "emoji_usage": emoji_count / total_posts,
            "question_frequency": question_count / total_posts,
            "response_style": sum(1 for post in posts if post.startswith('@')) / total_posts
        }

    def _calculate_detailed_metrics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive compatibility metrics"""
        
        def calculate_interest_similarity():
            user1_interests = context["user1"]["themes"]
            user2_interests = context["user2"]["themes"]
            common_categories = set(user1_interests.keys()) & set(user2_interests.keys())
            if not common_categories:
                return 0.0
            
            similarities = []
            for category in common_categories:
                similarity = 1 - abs(user1_interests[category] - user2_interests[category])
                similarities.append(similarity)
            
            return np.mean(similarities) * 100

        def calculate_communication_compatibility():
            style1 = context["user1"]["interaction_style"]
            style2 = context["user2"]["interaction_style"]
            
            factors = [
                1 - abs(style1["emoji_usage"] - style2["emoji_usage"]),
                1 - abs(style1["question_frequency"] - style2["question_frequency"]),
                1 - abs(style1["response_style"] - style2["response_style"])
            ]
            
            return np.mean(factors) * 100

        def calculate_emotional_compatibility():
            profile1 = context["user1"]["sentiment_profile"]
            profile2 = context["user2"]["sentiment_profile"]
            
            factors = [
                1 - abs(profile1["avg_polarity"] - profile2["avg_polarity"]),
                1 - abs(profile1["emotional_range"] - profile2["emotional_range"]),
                1 - abs(profile1["sentiment_stability"] - profile2["sentiment_stability"])
            ]
            
            return np.mean(factors) * 100

        # Calculate individual category scores
        interest_score = calculate_interest_similarity()
        communication_score = calculate_communication_compatibility()
        emotional_score = calculate_emotional_compatibility()
        
        # Calculate category-specific scores
        categories = {
            "romantic": self._calculate_romantic_score(context, interest_score, emotional_score),
            "friendship": self._calculate_friendship_score(context, interest_score, communication_score),
            "emotional": emotional_score,
            "communication": communication_score,
            "values": self._calculate_values_score(context),
            "lifestyle": self._calculate_lifestyle_score(context)
        }
        
        # Calculate overall score with weighted categories
        weights = {
            "romantic": 0.2,
            "friendship": 0.2,
            "emotional": 0.15,
            "communication": 0.15,
            "values": 0.15,
            "lifestyle": 0.15
        }
        
        overall_score = sum(score * weights[category] for category, score in categories.items())
        
        return {
            "overall": round(overall_score, 1),
            "categories": {k: round(v, 1) for k, v in categories.items()},
            "shared_interests": self._identify_shared_interests(context),
            "compatibility_factors": self._identify_compatibility_factors(context)
        }

    def _calculate_romantic_score(self, context: Dict[str, Any], interest_score: float, emotional_score: float) -> float:
        """Calculate romantic compatibility score"""
        factors = [
            interest_score * 0.3,  # Shared interests weight
            emotional_score * 0.4,  # Emotional compatibility weight
            self._calculate_activity_compatibility(context) * 0.3  # Activity patterns weight
        ]
        return sum(factors)

    def _calculate_friendship_score(self, context: Dict[str, Any], interest_score: float, communication_score: float) -> float:
        """Calculate friendship compatibility score"""
        factors = [
            interest_score * 0.4,  # Shared interests weight
            communication_score * 0.3,  # Communication style weight
            self._calculate_activity_compatibility(context) * 0.3  # Activity patterns weight
        ]
        return sum(factors)

    def _calculate_values_score(self, context: Dict[str, Any]) -> float:
        """Calculate values compatibility score"""
        # Analyze content themes and sentiment towards various topics
        user1_profile = self._analyze_value_indicators(context["user1"]["posts"])
        user2_profile = self._analyze_value_indicators(context["user2"]["posts"])
        
        return self._calculate_profile_similarity(user1_profile, user2_profile)

    def _calculate_lifestyle_score(self, context: Dict[str, Any]) -> float:
        """Calculate lifestyle compatibility score"""
        # Compare activity patterns and interests
        activity_compatibility = self._calculate_activity_compatibility(context)
        interest_alignment = self._calculate_interest_alignment(
            context["user1"]["interests"],
            context["user2"]["interests"]
        )
        
        return (activity_compatibility * 0.5 + interest_alignment * 0.5)

    def _generate_enhanced_narratives(self, context: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, str]:
        """Generate personalized compatibility narratives in a single API call"""
        try:
            # Create a combined prompt for all categories
            shared_interests = metrics["shared_interests"]
            compatibility_factors = metrics["compatibility_factors"]
            
            base_prompt = (
                f"Generate compatibility narratives for multiple categories between two users.\n"
                f"User 1: {context['user1']['display_name']} (Profile Summary: {context['user1']['description']}, Posts: {context['user1']['posts']})\n"
                f"User 2: {context['user2']['display_name']} (Profile Summary: {context['user2']['description']}, Posts: {context['user2']['posts']})\n"
                f"Shared Interests: {', '.join(shared_interests)}\n"
                f"Key Compatibility Factors: {', '.join(compatibility_factors)}\n\n"
                "Guidelines:\n"
                "- Adopt a conversational and friendly tone\n"
                "- Use engaging, relatable language\n"
                "- Add depth to each narrative\n"
                "- Highlight both positive and constructive aspects\n"
                "- Emphasize shared interests and unique traits\n"
                "- Provide actionable insights\n"
                "- DO NOT quote specific posts\n"
                "- DO NOT use the phrases 'tech-savvy' or 'shared love of technology' as they lack depth and are repetitive\n"
                "- Keep each narrative within 150 words\n\n"
                "Your entire response/output is going to consist of a single JSON object {}, and you MUST NOT wrap it within JSON md markers.\n"
                "Return a JSON object with the following structure:\n"
                "{\n"
                '  "romantic": "narrative for romantic compatibility",\n'
                '  "friendship": "narrative for friendship compatibility",\n'
                '  "emotional": "narrative for emotional compatibility",\n'
                '  "communication": "narrative for communication compatibility",\n'
                '  "values": "narrative for values compatibility",\n'
                '  "lifestyle": "narrative for lifestyle compatibility"\n'
                "}\n\n"
                "Include scores in narratives:\n"
                f"Romantic Score: {metrics['categories']['romantic']}%\n"
                f"Friendship Score: {metrics['categories']['friendship']}%\n"
                f"Emotional Score: {metrics['categories']['emotional']}%\n"
                f"Communication Score: {metrics['categories']['communication']}%\n"
                f"Values Score: {metrics['categories']['values']}%\n"
                f"Lifestyle Score: {metrics['categories']['lifestyle']}%"
            )

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-001:generateContent?key={self._get_random_api_key()}"
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": base_prompt
                    }]
                }],
                "safetySettings": [{
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                }]
            }
            
            response = requests.post(url, json=data)
            
            if response.status_code == 200:
                response_data = response.json()
                if 'candidates' in response_data and response_data['candidates']:
                    narrative_text = response_data['candidates'][0]['content']['parts'][0]['text']
                    
                    # Extract JSON from the response
                    try:
                        # Find the JSON object in the response text
                        json_start = narrative_text.find('{')
                        json_end = narrative_text.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = narrative_text[json_start:json_end]
                            narratives = json.loads(json_str)
                            
                            # Verify all required categories are present
                            required_categories = ["romantic", "friendship", "emotional", "communication", "values", "lifestyle"]
                            for category in required_categories:
                                if category not in narratives:
                                    narratives[category] = f"Failed to generate {category} narrative."
                                    
                            return narratives
                    except json.JSONDecodeError:
                        print("Failed to parse JSON from response")
                        
            # Fallback responses if API call fails
            return {
                "romantic": f"Failed to generate romantic compatibility narrative. Score: {metrics['categories']['romantic']}%",
                "friendship": f"Failed to generate friendship compatibility narrative. Score: {metrics['categories']['friendship']}%",
                "emotional": f"Failed to generate emotional compatibility narrative. Score: {metrics['categories']['emotional']}%",
                "communication": f"Failed to generate communication compatibility narrative. Score: {metrics['categories']['communication']}%",
                "values": f"Failed to generate values compatibility narrative. Score: {metrics['categories']['values']}%",
                "lifestyle": f"Failed to generate lifestyle compatibility narrative. Score: {metrics['categories']['lifestyle']}%"
            }
        except Exception as e:
            print(f"Error generating narratives: {str(e)}")
            return {category: f"Error generating narrative: {str(e)}" for category in metrics['categories']}

    def _generate_narrative_with_ai(self, prompt: str) -> str:
        """
        This method is now deprecated as narratives are generated in batch
        Kept for backward compatibility
        """
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-001:generateContent?key={self.GOOGLE_API_KEY}"
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "safetySettings": [{
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                }]
            }
            
            response = requests.post(url, json=data)
            
            if response.status_code == 200:
                response_data = response.json()
                if 'candidates' in response_data and response_data['candidates']:
                    return response_data['candidates'][0]['content']['parts'][0]['text']
            
            return f"Failed to generate narrative. Status code: {response.status_code}"
                        
        except Exception as e:
            return f"Narrative generation error: {str(e)}"

    def _generate_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate an engaging summary of the compatibility analysis
        
        Args:
            metrics (Dict[str, Any]): Dictionary containing overall score, category scores,
                                    shared interests, and compatibility factors
        
        Returns:
            str: A detailed, engaging summary of the compatibility analysis
        """
        score = metrics["overall"]
        categories = metrics["categories"]
        shared_interests = metrics["shared_interests"]
        
        # Determine compatibility level
        compatibility_level = self._get_compatibility_level(score)
        
        # Find strongest and weakest categories
        strongest_category = max(categories.items(), key=lambda x: x[1])
        weakest_category = min(categories.items(), key=lambda x: x[1])
        
        # Build summary components
        opening = self._get_opening_statement(score, compatibility_level)
        strengths = self._format_strengths(strongest_category, shared_interests)
        areas_for_growth = self._format_areas_for_growth(weakest_category)
        
        # Combine all components into final summary
        summary = f"{opening}\n\n{strengths}\n\n{areas_for_growth}"
        
        if score >= 75:
            summary += "\n\nOverall, this appears to be a highly promising match!"
        elif score >= 50:
            summary += "\n\nThere's good potential here, though some areas may benefit from open communication and understanding."
        else:
            summary += "\n\nWhile there may be some challenges, every connection offers opportunities for growth and learning."
        
        return summary

    def _get_compatibility_level(self, score: float) -> str:
        """Determine the compatibility level based on the overall score"""
        if score >= 80:
            return "exceptional"
        elif score >= 70:
            return "strong"
        elif score >= 60:
            return "good"
        elif score >= 50:
            return "moderate"
        else:
            return "developing"

    def _get_opening_statement(self, score: float, level: str) -> str:
        """Generate an appropriate opening statement based on score and level"""
        return (f"Based on our comprehensive analysis, these profiles show a {level} "
                f"compatibility level with an overall score of {score}%.")

    def _format_strengths(self, strongest_category: Tuple[str, float], 
                        shared_interests: List[str]) -> str:
        """Format the strengths section of the summary"""
        category_name, category_score = strongest_category
        
        strengths = f"The strongest area of compatibility is {category_name} "
        strengths += f"({category_score:.1f}%). "
        
        if shared_interests:
            if len(shared_interests) == 1:
                strengths += f"There is a shared interest in {shared_interests[0]}."
            else:
                interests_text = ", ".join(shared_interests[:-1]) + f" and {shared_interests[-1]}"
                strengths += f"Common interests include {interests_text}."
        
        return strengths

    def _format_areas_for_growth(self, weakest_category: Tuple[str, float]) -> str:
        """Format the areas for growth section of the summary"""
        category_name, category_score = weakest_category
        
        growth_statements = {
            "romantic": "There may be room for developing deeper emotional connections.",
            "friendship": "Building more shared experiences could strengthen the friendship.",
            "emotional": "Working on emotional understanding could enhance the connection.",
            "communication": "More open and consistent communication might be beneficial.",
            "values": "Taking time to understand each other's perspectives could help bridge any gaps.",
            "lifestyle": "Finding common ground in daily routines and activities could improve compatibility."
        }
        
        return (f"The area that might benefit from more attention is {category_name} "
                f"({category_score:.1f}%). {growth_statements.get(category_name, '')}")
    
    def _analyze_activity_pattern(self, posts: List[Dict]) -> Dict[str, float]:
        """Analyze user's posting patterns and activity preferences."""
        if not posts:
            return {
                "posting_frequency": 0.0,
                "time_distribution": {},
                "consistency_score": 0.0,
                "engagement_level": 0.0
            }
        
        # Extract timestamps and convert to datetime
        timestamps = [
            datetime.fromisoformat(post.get('post', {}).get('record', {}).get('createdAt', '').replace('Z', '+00:00'))
            for post in posts if post.get('post', {}).get('record', {}).get('createdAt')
        ]
        
        if not timestamps:
            return {
                "posting_frequency": 0.0,
                "time_distribution": {},
                "consistency_score": 0.0,
                "engagement_level": 0.0
            }
        
        # Calculate posting frequency (posts per day)
        date_range = max(timestamps) - min(timestamps)
        posts_per_day = len(timestamps) / (date_range.days + 1) if date_range.days >= 0 else 0
        
        # Analyze time distribution
        hours = [ts.hour for ts in timestamps]
        hour_distribution = Counter(hours)
        total_posts = len(hours)
        time_distribution = {hour: count/total_posts for hour, count in hour_distribution.items()}
        
        # Calculate consistency score
        hour_variance = np.var(list(time_distribution.values())) if time_distribution else 1.0
        consistency_score = 1.0 - min(hour_variance, 1.0)
        
        # Calculate engagement level based on post frequency and consistency
        engagement_level = (posts_per_day * 0.6 + consistency_score * 0.4) * 100
        
        return {
            "posting_frequency": posts_per_day,
            "time_distribution": time_distribution,
            "consistency_score": consistency_score,
            "engagement_level": engagement_level
        }

    def _extract_interests(self, posts: List[str]) -> List[str]:
        """Extract user interests from their posts using NLP techniques."""
        if not posts:
            return []
        
        # Combine all posts
        text = ' '.join(posts).lower()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Calculate word frequencies
        word_freq = Counter(tokens)
        
        # Map words to interest categories
        interest_scores = defaultdict(float)
        for word, count in word_freq.items():
            for category, keywords in self.interest_categories.items():
                if any(keyword in word for keyword in keywords):
                    interest_scores[category] += count
        
        # Filter top interests (those with scores above mean)
        mean_score = np.mean(list(interest_scores.values())) if interest_scores else 0
        top_interests = [category for category, score in interest_scores.items() 
                        if score > mean_score]
        
        return sorted(top_interests)


    def _identify_shared_interests(self, context: Dict[str, Any]) -> List[str]:
        """Identify common interests between two users."""
        user1_interests = set(context["user1"]["interests"])
        user2_interests = set(context["user2"]["interests"])
        return sorted(list(user1_interests & user2_interests))

    def _identify_compatibility_factors(self, context: Dict[str, Any]) -> List[str]:
        """Identify key factors contributing to compatibility."""
        factors = []
        
        # Compare interaction styles
        style1 = context["user1"]["interaction_style"]
        style2 = context["user2"]["interaction_style"]
        
        if abs(style1["emoji_usage"] - style2["emoji_usage"]) < 0.2:
            factors.append("similar communication style")
        
        # Compare sentiment profiles
        sent1 = context["user1"]["sentiment_profile"]
        sent2 = context["user2"]["sentiment_profile"]
        
        if abs(sent1["avg_polarity"] - sent2["avg_polarity"]) < 0.3:
            factors.append("emotional alignment")
            
        # Compare activity patterns
        if self._calculate_activity_compatibility(context) > 0.7:
            factors.append("compatible schedules")
            
        # Add shared interests if significant
        shared = self._identify_shared_interests(context)
        if len(shared) > 2:
            factors.append("multiple shared interests")
            
        return factors

    def _analyze_value_indicators(self, posts: List[str]) -> Dict[str, float]:
        """Analyze indicators of personal values from posts."""
        if not posts:
            return {}
            
        # Combine posts
        text = ' '.join(posts).lower()
        
        # Define value categories and associated keywords
        value_categories = {
            'community': ['community', 'together', 'support', 'help', 'share', 'collaborate', 'team', 'bond'],
            'achievement': ['goal', 'success', 'achieve', 'accomplish', 'win', 'milestone', 'award', 'celebrate'],
            'growth': ['learn', 'improve', 'grow', 'develop', 'progress', 'self-care', 'evolve', 'better'],
            'creativity': ['create', 'design', 'imagine', 'innovate', 'original', 'craft', 'express', 'art'],
            'harmony': ['peace', 'balance', 'calm', 'harmony', 'understanding', 'acceptance', 'meditate', 'zen'],
            'adventure': ['adventure', 'explore', 'travel', 'discover', 'journey', 'wanderlust', 'trip', 'experience'],
            'sustainability': ['sustain', 'eco', 'green', 'recycle', 'environment', 'nature', 'climate', 'planet'],
            'leadership': ['lead', 'inspire', 'guide', 'mentor', 'direct', 'vision', 'empower', 'influence'],
            'positivity': ['happy', 'joy', 'smile', 'positive', 'kindness', 'grateful', 'blessing', 'cheerful'],
            'relationships': ['family', 'friends', 'love', 'bond', 'connection', 'relationship', 'partner', 'network'],
            'activism': ['justice', 'rights', 'protest', 'equality', 'freedom', 'advocate', 'change', 'reform'],
            'technology': ['tech', 'innovation', 'future', 'software', 'AI', 'device', 'app', 'digital'],
            'humor': ['funny', 'joke', 'laugh', 'meme', 'hilarious', 'giggle', 'comedy', 'wit'],
            'health': ['fitness', 'exercise', 'diet', 'healthy', 'wellness', 'mental', 'strength', 'workout'],
            'sex': ['sexual', 'tits', 'titties', 'meat', 'dick', 'cum', 'anal', 'ass', 'plug', 'nudes', 'nsfw', '#afterdark', 'skeetsafterdark', 'pussy', 'squirt', 'sex', 'boobs']
        }

        
        # Calculate value scores
        scores = {}
        for category, keywords in value_categories.items():
            score = sum(text.count(keyword) for keyword in keywords)
            scores[category] = score if score > 0 else 0.0
            
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
            
        return scores

    def _calculate_profile_similarity(self, profile1: Dict[str, float], 
                                    profile2: Dict[str, float]) -> float:
        """Calculate similarity between two value profiles."""
        if not profile1 or not profile2:
            return 0.0
            
        # Get common categories
        common_categories = set(profile1.keys()) & set(profile2.keys())
        if not common_categories:
            return 0.0
            
        # Calculate cosine similarity
        vec1 = [profile1[cat] for cat in common_categories]
        vec2 = [profile2[cat] for cat in common_categories]
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return similarity * 100


    def _calculate_activity_compatibility(self, context: Dict[str, Any]) -> float:
        """Calculate compatibility of users' activity patterns."""
        pattern1 = context["user1"]["activity_pattern"]
        pattern2 = context["user2"]["activity_pattern"]
        
        if not pattern1 or not pattern2:
            return 0.0
        
        # Compare time distributions
        time_dist1 = pattern1["time_distribution"]
        time_dist2 = pattern2["time_distribution"]
        
        # Calculate overlap in active hours
        all_hours = set(time_dist1.keys()) | set(time_dist2.keys())
        overlap_score = 0.0
        
        for hour in all_hours:
            score1 = time_dist1.get(hour, 0)
            score2 = time_dist2.get(hour, 0)
            overlap_score += min(score1, score2)
        
        # Compare engagement levels
        engagement_diff = abs(pattern1["engagement_level"] - pattern2["engagement_level"]) / 100
        engagement_compatibility = 1 - engagement_diff
        
        # Calculate final score weighted between time overlap and engagement
        return (overlap_score * 0.6 + engagement_compatibility * 0.4) * 100

    def _calculate_interest_alignment(self, interests1: List[str], 
                                    interests2: List[str]) -> float:
        """Calculate alignment between two sets of interests."""
        if not interests1 or not interests2:
            return 0.0
        
        set1 = set(interests1)
        set2 = set(interests2)
        
        # Calculate Jaccard similarity
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        # Weight shared interests more heavily
        base_similarity = intersection / union
        bonus = 0.2 if intersection >= 3 else 0.1 if intersection >= 2 else 0
        
        return min((base_similarity + bonus) * 100, 100)


