"""
Simulated Patient Agent for evaluation.

This agent simulates a patient responding to IIA (Intake Interview Agent) questions
by extracting relevant symptom information from ETEK vignettes.
"""
import os
import logging
from typing import List, Dict, Any

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger("PatientAgent")

PATIENT_SYSTEM_PROMPT = """  
You are participating in a healthcare workflow research simulation, portraying a patient in an Emergency Department triage interview.

### SIMULATION CONTEXT
You are role-playing a patient based on the clinical scenario provided below. This simulation helps evaluate AI-assisted triage systems for research purposes.

### YOUR SCENARIO
{scenario_description}

### RESPONSE GUIDELINES
1. **Respond naturally**: Use conversational, everyday language as a real patient would. Include natural speech patterns like "um", expressions of concern, or uncertainty.
2. **Stay in character**: Base all responses on the scenario details provided. Be consistent with the information given.
3. **Answer what is asked**: Provide information relevant to the question. Avoid volunteering extra details unless prompted.
4. **Handle unknown information**: If asked about something not in the scenario, respond with "I'm not sure" or "I don't think so."
5. **Clinical measurements**: Do not mention vital signs (heart rate, blood pressure, temperature, oxygen levels, respiratory rate) unless specifically asked. These are measured by clinical staff, not reported by patients.
6. **Severity ratings**: If asked about pain or symptom severity, use the scenario information or provide a reasonable estimate consistent with the described condition.
7. **Keep responses brief**: Typically 1-3 sentences per response.

### CHARACTER NOTES
- You are the patient, not a healthcare professional.
- Describe symptoms in everyday terms (e.g., "my side really hurts" rather than clinical terminology).
- Express appropriate emotions and concerns a patient might have.
"""


def get_patient_llm():
    """Initialize the patient simulation LLM."""
    return AzureChatOpenAI(
        azure_deployment=os.environ.get("AZURE_OPENAI_GENERAL_DEPLOYMENT_NAME", "gpt-4.1-mini"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        temperature=0.7,  # Some variation for naturalistic responses
    )


class PatientAgent:
    """Simulated patient that responds to IIA questions based on ETEK vignette."""
    
    def __init__(self, scenario_description: str):
        """
        Initialize patient agent with scenario context.
        
        Args:
            scenario_description: The ETEK vignette describing the patient's condition
        """
        self.scenario = scenario_description
        self.llm = get_patient_llm()
        self.conversation_history: List[Dict[str, str]] = []
        
        # Prepare system message with scenario
        self.system_message = SystemMessage(
            content=PATIENT_SYSTEM_PROMPT.format(scenario_description=scenario_description)
        )
    
    def respond(self, nurse_message: str) -> str:
        """
        Generate a patient response to the nurse/IIA message.
        
        Args:
            nurse_message: The question or statement from IIA
            
        Returns:
            Patient's response as a string
        """
        # Build message list
        messages = [self.system_message]
        
        # Add conversation history
        for turn in self.conversation_history:
            if turn["role"] == "nurse":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))
        
        # Add current nurse message
        messages.append(HumanMessage(content=nurse_message))
        
        # Generate response
        response = self.llm.invoke(messages)
        patient_reply = response.content
        
        # Update history
        self.conversation_history.append({"role": "nurse", "content": nurse_message})
        self.conversation_history.append({"role": "patient", "content": patient_reply})
        
        logger.debug(f"Patient response: {patient_reply[:100]}...")
        return patient_reply
    
    def get_transcript(self) -> List[Dict[str, str]]:
        """Return the full conversation transcript."""
        return self.conversation_history.copy()
    
    def reset(self):
        """Clear conversation history for a new interview."""
        self.conversation_history = []


# Standalone test
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test with scenario 1
    test_scenario = """Joe is 47 years of age; he presents to the ED with a letter of referral from his local doctor. 
    He has left flank pain radiating to his groin. His local doctor has given him oxycodone 5 mg with little effect. 
    He has a history of left renal colic. His respiratory rate is 26 breaths per minute, his heart rate is 120 beats 
    per minute and his blood pressure is 128/78 mmHg. His skin is cool and clammy. He rates his pain as 8/10."""
    
    patient = PatientAgent(test_scenario)
    
    # Simulate interview
    questions = [
        "Hello, what brings you to the emergency department today?",
        "When did this pain start?",
        "On a scale of 1 to 10, how would you rate your pain?",
        "Does the pain go anywhere else, like down your leg or to your back?"
    ]
    
    print("=== Patient Agent Test ===\n")
    for q in questions:
        print(f"Nurse: {q}")
        response = patient.respond(q)
        print(f"Patient: {response}\n")
