import os, time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load API keys from .env file
client = OpenAI()  # Uses OPENAI_API_KEY from environment

class AuditHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("proposed_plan.md"):
            print("🔍 New Plan Detected. Analyzing for drift...")
            with open("proposed_plan.md", "r") as f:
                plan = f.read()
            
            # Send to GPT-4o for Visual/Logic Audit
            audit_query = f"""Analyze this implementation plan and generate a comprehensive drift report.

PART 1: VISUAL IMPACT DIAGRAM
Generate a clear, visually distinct Mermaid.js diagram showing:

1. Use different shapes and colors to distinguish:
   - 🆕 NEW files: Large rectangles with GREEN fill (#90EE90) and bold borders
   - ✏️ MODIFIED files: Hexagons with YELLOW fill (#FFE4B5) and dashed borders
   - 📦 NEW functions/classes: Small rounded rectangles with LIGHT BLUE (#ADD8E6)
   - 📤 NEW outputs: Cylinders with PURPLE (#E6E6FA)

2. Use clear visual hierarchy with subgraphs:
   - Group "NEW CODE" separately
   - Group "MODIFIED CODE" separately
   - Group "OUTPUTS" separately
   - Show data flow with labeled arrows

3. Keep cognitive load LOW:
   - Maximum 3-4 items per subgraph
   - Use clear, concise labels (max 3 words)

Example Mermaid structure:
```mermaid
graph TB
    subgraph NEW["🆕 NEW CODE"]
        A["serve_receive.py<br/>NEW FILE"]
        B("ServeReceiveDetector<br/>NEW CLASS")
    end
    subgraph MOD["✏️ MODIFIED CODE"]
        D{{"pipeline.py<br/>_export_results<br/>MODIFIED"}}
    end
    subgraph OUT["📤 OUTPUTS"]
        F[("serve_receive.jsonl<br/>NEW")]
        G[("serve_receive.csv<br/>NEW")]
    end
    D -->|calls| B
    B -->|creates| F
    B -->|creates| G
    classDef newFile fill:#90EE90,stroke:#333,stroke-width:3px
    classDef modFile fill:#FFE4B5,stroke:#333,stroke-width:2px,stroke-dasharray:5 5
    classDef output fill:#E6E6FA,stroke:#333,stroke-width:2px
    class A,B newFile
    class D modFile
    class F,G output
```

PART 2: END RESULT OUTPUTS
After the Mermaid diagram, add a section titled "## 📊 Expected Outputs" that shows:

1. **What files will be created/modified** (list all output files)
2. **Sample output examples** (show realistic CSV/JSON examples based on the feature)
3. **How to use the outputs** (what can the user do with these files)

Format this section as:
```markdown
## 📊 Expected Outputs

### New Files Generated
- `filename.ext` - Description of what this contains

### Sample Output Examples

**filename.csv:**
```csv
Header1,Header2,Header3
value1,value2,value3
```

**filename.json:**
```json
{{"field": "value"}}
```

### How to Use
- Action 1: Description
- Action 2: Description
```

Now analyze THIS plan:

{plan}

Generate both the Mermaid diagram AND the Expected Outputs section."""
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": audit_query}]
            )

            with open("drift_report.md", "w") as f:
                f.write(response.choices[0].message.content)
            print("✅ Drift Report Generated in drift_report.md")

observer = Observer()
observer.schedule(AuditHandler(), ".", recursive=False)
observer.start()
try:
    while True: time.sleep(1)
except KeyboardInterrupt: observer.stop()
