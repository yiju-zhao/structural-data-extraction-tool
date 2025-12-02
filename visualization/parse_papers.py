import json
import re

def parse_papers(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by the separator line
    raw_papers = content.split('--------------------------------------------------------------------------------')
    
    papers = []
    
    for raw_paper in raw_papers:
        if not raw_paper.strip():
            continue
            
        paper = {}
        lines = raw_paper.strip().split('\n')
        
        bibtex_lines = []
        in_bibtex = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Title: '):
                paper['title'] = line[7:]
            elif line.startswith('Authors: '):
                paper['authors'] = line[9:]
            elif line.startswith('Venue: '):
                paper['venue'] = line[7:]
            elif line.startswith('Affinity Score: '):
                try:
                    paper['affinity_score'] = float(line[16:])
                except:
                    paper['affinity_score'] = 0.0
            elif line.startswith('Link: '):
                paper['link'] = line[6:]
            elif line.startswith('Keywords: '):
                paper['keywords'] = line[10:]
            elif line.startswith('BibTeX:'):
                in_bibtex = True
                continue
            
            if in_bibtex:
                bibtex_lines.append(line)
        
        if bibtex_lines:
            paper['bibtex'] = '\n'.join(bibtex_lines)
            
        if 'title' in paper: # Ensure at least a title exists
            papers.append(paper)
            
    return papers

if __name__ == "__main__":
    input_file = "physical_ai/neurips_2025_physical_ai.txt" 
    output_file = "physical_ai/papers.json"
    
    try:
        papers = parse_papers(input_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        print(f"Successfully parsed {len(papers)} papers to {output_file}")
    except Exception as e:
        print(f"Error parsing papers: {e}")
