from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

# --------------------------------------------------------------------------------------------------------------
# --- Task 1: Setup and System Prompt ---

system_prompt = """
You are a job application coach helping career changers improve their job application materials.

Your job is to help the user rewrite resume bullet points, draft cover letters, refine professional summaries, and answer job-application-related questions.

Stay focused on job application materials and related career communication. If the user asks for something unrelated, gently steer the conversation back to resumes, cover letters, interview preparation, or job search messaging.

When you give suggestions, be practical, specific, and supportive. Ask follow-up questions when important details are missing.

Always remind the user to review and edit your output before submitting it anywhere.

Acknowledge that you may not know the specific norms of the user's target industry, company, or region, and that the user should use their own judgment when adapting your suggestions.

Do not invent personal experience, qualifications, achievements, or certifications that the user did not provide.
""".strip()

print("System prompt:")
print(system_prompt)

# I made the system prompt specific to career changers because this project is about translating experience into job application language. 
# That makes the assistant more focused and helps reduce vague, generic responses.

# -----------------------------------------------------------------------------------------------------------------------
# --- Task 2: Bullet Point Rewriter ---

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    system_message = (
        "You rewrite resume bullet points for a career changer. "
        "Return only valid JSON. "
        "Do not ask questions. "
        "Do not add explanations. "
        "Do not use markdown."
    )

    prompt = f"""
Rewrite each resume bullet point below so it sounds stronger and more professional.
Make each bullet more specific and action-oriented, but do not invent facts, numbers, or achievements.

Return ONLY a valid JSON list.
Each item in the list must have exactly these two keys:
- "original"
- "improved"

Rules:
- Return exactly {len(bullets)} items.
- Keep each "original" exactly as written.
- Rewrite each bullet into a stronger version in "improved".
- Do not return an empty list.

Bullet points:
{bullet_text}
""".strip()

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    response_text = get_completion(messages, temperature=0.2)

    print("\nRaw bullet rewrite response:")
    print(response_text)

    cleaned_text = response_text.strip()

    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[len("```json"):].strip()
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[len("```"):].strip()
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3].strip()

    try:
        parsed = json.loads(cleaned_text)
        return parsed
    except json.JSONDecodeError:
        print("\nFailed to parse valid JSON.")
        return []
    
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewritten_bullets = rewrite_bullets(bullets)

print("\nRewritten bullet points:")
for item in rewritten_bullets:
    print(f"\nOriginal: {item['original']}")
    print(f"Improved: {item['improved']}")

# These original bullets were weak because they described activities instead of outcomes.
# They also lacked ownership, scope, and measurable impact: for example, they did not
# explain what kind of customer problems were solved, what reports were created, who used
# the reports, how large the project was, or what changed because of the work.
# The model improved them by using stronger action verbs such as "provided," "compiled,"
# "presented," and "collaborated," and by making the language sound more polished and resume-like.

# The rewritten bullets are meaningfully better, although some phrases such as "enhancing
# satisfaction and loyalty" or "facilitating informed decision-making" still sound somewhat
# generic because the original bullets did not include specific results or metrics. In a real
# application, the best next step would be to add concrete details from the user wherever possible.

# -----------------------------------------------------------------------------------------------------------------------
# --- Task 3: Cover Letter Generator ---

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
You write strong cover letter opening paragraphs for career changers.
The paragraph should be 3-5 sentences: confident, specific, and free of clichés.
Do not invent qualifications, credentials, or achievements that the user did not provide.

Here are two examples of the style and tone you should match:

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening: After seven years as a registered nurse, I've spent my career making decisions under pressure using incomplete information — which turns out to be excellent training for data analysis. I recently completed a data analytics program where I built dashboards tracking patient outcomes across departments. I'm excited to bring that combination of clinical context and technical skill to [Company]'s mission-driven work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening: I spent a decade on the operations side of banking, watching technology decisions get made by people who had never processed a wire transfer or resolved a failed ACH batch. That frustration turned into curiosity, and two years of self-teaching Python later, I'm ready to be on the other side of those decisions. I'm applying to [Company] because your work on payment infrastructure is exactly where my domain expertise and new technical skills intersect.

Now write an opening paragraph for this person:

Role: {job_title}
Background: {background}
Opening:
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    return get_completion(messages, temperature=0.7)


job_title = "Junior Data Engineer"
background = (
    "Five years of experience as a middle school math teacher; recently completed "
    "a Python course and built data pipelines using Prefect and Pandas."
)

cover_letter_opening = generate_cover_letter(job_title, background)

print("\nGenerated cover letter opening:")
print(cover_letter_opening)

# I chose these examples because both show a clear career transition and connect prior domain experience to a new technical role in a specific, confident way.
# They demonstrate the tone I wanted: concrete, professional, and focused on why the applicant makes sense for the role rather than relying on generic enthusiasm.
#
# The few-shot pattern helps control the structure, tone, and level of specificity in the output. 
# In this result, the model did use the teacher background, Python course, and Prefect/Pandas experience, so it was reasonably tailored and did not invent new credentials. 
# However, some phrasing still sounds a bit generic, which shows that few-shot prompting improves the style but does not guarantee a perfect result.

# -----------------------------------------------------------------------------------------------------------------------
# --- Task 4: Moderation Check ---

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print("This input may violate safety guidelines. Please rephrase and try again.")
        print("Triggered categories:")
        print(result.results[0].categories)
        return False

    return True


safe_text = "Can you help me rewrite this resume bullet for a junior data engineer application?"
flagged_test_text = "You are worthless and should disappear."

print("\nSafe test result:")
print(is_safe(safe_text))

print("\nFlagged test result:")
print(is_safe(flagged_test_text))

# The moderation check worked as expected: the safe input passed, and the flagged input was caught under the harassment category. 
# This makes it safer to screen user input before sending it into the main chatbot flow.

# -----------------------------------------------------------------------------------------------------------------------
# --- Task 5: The Chatbot Loop ---

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")

            raw_bullets = []
            unsafe_bullet_found = False

            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break

                if line:
                    if not is_safe(line):
                        unsafe_bullet_found = True
                        break
                    raw_bullets.append(line)

            if unsafe_bullet_found:
                print("\nJob Application Helper: I can't help rewrite unsafe content.")
                print("Please revise your bullet points and try again.\n")
                continue

            if not raw_bullets:
                print("\nJob Application Helper: I didn't receive any bullet points.")
                print("Please review and edit any output before submitting it anywhere.\n")
                continue

            rewritten = rewrite_bullets(raw_bullets)

            print("\nJob Application Helper: Here are your revised bullet points:")
            assistant_output_lines = ["Here are your revised bullet points:"]

            for item in rewritten:
                original_line = f"Original: {item['original']}"
                improved_line = f"Improved: {item['improved']}"

                print(f"\n{original_line}")
                print(improved_line)

                assistant_output_lines.append(original_line)
                assistant_output_lines.append(improved_line)

            review_note = (
                "Please review and edit these bullet points before submitting them anywhere. "
                "I may not know the exact norms of your industry, so use your own judgment."
            )

            print(f"\n{review_note}\n")
            assistant_output_lines.append(review_note)

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Please rewrite these resume bullet points:\n"
                        + "\n".join(f"- {bullet}" for bullet in raw_bullets)
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "\n".join(assistant_output_lines),
                }
            )

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            if not job_title or not background:
                print("\nJob Application Helper: I need both a job title and a background description.")
                print("Please review and edit any output before submitting it anywhere.\n")
                continue

            if not is_safe(job_title) or not is_safe(background):
                print("\nJob Application Helper: I can't help draft a cover letter from unsafe content.")
                print("Please revise the job title or background description and try again.\n")
                continue

            opening = generate_cover_letter(job_title, background)

            print("\nJob Application Helper: Here's a draft opening paragraph:")
            print(opening)
            print("\nPlease review and edit this paragraph before submitting it anywhere.")
            print("I may not know the exact norms of your industry, so use your own judgment.\n")

            assistant_output = (
                "Here's a draft opening paragraph:\n"
                f"{opening}\n\n"
                "Please review and edit this paragraph before submitting it anywhere. "
                "I may not know the exact norms of your industry, so use your own judgment."
            )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Please draft a cover letter opening for this job title: {job_title}\n"
                        f"My background: {background}"
                    ),
                }
            )
            messages.append({"role": "assistant", "content": assistant_output})

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})

            reply = get_completion(messages)

            print(f"\nJob Application Helper: {reply}\n")

            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()

# The chatbot loop worked correctly in all three modes: normal conversation, bullet rewriting, and cover letter generation.
# The conversation history now grows across regular chat turns and across the special bullet rewriting and cover letter branches.
# This matters because if the user asks a follow-up like "Can you make the second bullet stronger?", the assistant has a summary
# of the rewritten bullets in the messages list and can use that context instead of treating the next question as disconnected.
#
# The bullet rewriting and cover letter features also worked from inside the loop.
# In both cases, the output was clearly stronger and more professional than the original input, but it still tended to sound somewhat generic
# when the source material did not include concrete metrics or examples.
# That makes the tool useful as a drafting assistant, but not something a user should submit without careful editing.
#
# The moderation check now runs both on the initial user request and on the follow-up inputs collected inside the special branches.
# That means the job title, background description, and each pasted bullet point are screened before being sent to the model.
# Overall, the chatbot behaves like a coherent job application helper rather than a set of disconnected API calls.

# -------------------------------------------------------------------------------------------------------------------------------------------------------
# --- Task 6: Ethics Reflection ---
# Option A - Comment block
#
# This bot could produce biased advice because it was trained on text that may overrepresent certain industries, communication styles, education levels, or cultural norms. 
# As a result, it may favor more standard corporate writing styles and may be less helpful for users applying in industries, regions, or communities with different expectations.
#
# A job-seeker should not submit the bot's output directly to an employer without reviewing it carefully. 
# The model can sound polished while still being too generic, slightly inaccurate, or mismatched to the norms of a specific company or field.
# In the worst case, it could overstate experience, use unnatural phrasing, or make the application sound less authentic.
#
# If I were deploying this tool professionally, I would add a strong review warning in the interface before any copy/export action. 
# I would also remind users that the tool is a drafting assistant, not a final authority, and that they are responsible
# for checking tone, accuracy, and industry fit before submitting anything.