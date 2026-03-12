class MetadataService:
    def __init__(self, llm):
        self.llm = llm

    def generate_title(self, text: str) -> str:
        prompt = f"Generate a concise and descriptive title for the following content:\n{text[:500]}"
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating title: {e}")
            return "Unknown Title"

    def generate_keywords(self, text: str) -> list[str]:
        prompt = f"Extract 5-10 keywords from the following content:\n{text[:500]}"
        try:
            resp = self.llm.complete(prompt)

            if resp.text.startswith("Keywords:"):
                lines = resp.text.split("\n")[1:]
                keywords = [line.strip("- ").strip() for line in lines if line.strip()]
            elif "," in resp.text:
                keywords = [kw.strip() for kw in resp.text.split(",")]
            else:
                keywords = resp.text.split()

            return list(set(kw for kw in keywords if kw))
        except Exception as e:
            print(f"Error generating keywords: {e}")
            return ["No keywords available"]

    def generate_summary(self, text: str) -> str:
        prompt = f"Summarize the following content in 2-3 sentences:\n{text[:1000]}"
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "No summary available."
