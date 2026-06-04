export function createParsePrompt(resumeText: string): string {
  return `
Extract the following information from the given resume in a strictly structured JSON format.
Ensure all fields are present, even if empty. Use double quotes for all JSON keys and string values.

JSON Structure:
{
  "Name": "Full Name",
  "Contact": { "Email": "email@example.com", "Phone": "(123) 456-7890" },
  "Summary": "Brief professional summary",
  "Skills": [ {"Skill": "Skill Name", "Proficiency": "Expert/Advanced/Intermediate/Basic"} ],
  "Work_Experience": [ { "Company": "Company Name", "Job_Title": "Job Title", "Duration": "MM-YYYY to MM-YYYY", "Location": "City, State", "Achievements": [ {"Description": "Achievement description", "Metrics": "Quantifiable metric"} ] } ],
  "Education": [ { "Degree": "Degree Name", "Institution": "Institution Name", "Year": "MM-YYYY" } ],
  "Projects": [ { "Name": "Project Name", "Description": "Brief project description", "Impact": "Quantifiable impact" } ],
  "Certifications": [ "Certification Name" ],
  "Core_Competencies": [ "Competency 1", "Competency 2" ],
  "Key_Achievements": [ { "Description": "Achievement description", "Metrics": "Quantifiable metric" } ]
}

Resume text to parse:
---
${resumeText}
---

Provide only valid JSON matching the structure above. No markdown fences.
`.trim();
}

export function createRepairPrompt(resumeText: string, badResponse: string): string {
  const clipped = badResponse.slice(0, 8000);
  const resumeClip = resumeText.slice(0, 12000);
  return `The following text was supposed to be a single JSON object for a resume parse but may be invalid or wrapped in markdown.
Fix it: output ONLY one valid JSON object with the same schema as before (Name, Contact, Summary, Skills, Work_Experience, Education, Projects, Certifications, Core_Competencies, Key_Achievements).

Broken output:
${clipped}

Original resume (for reference):
---
${resumeClip}
---
Output only the JSON object, no other text.`;
}
