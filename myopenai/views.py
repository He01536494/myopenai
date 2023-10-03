from django.shortcuts import render
#from . import hello
from django.http import HttpResponse
from django.contrib import admin

# imports
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial
import mwclient 
import mwparserfromhell
import re  
from IPython.display import display


# Create your views here.
def index(request):
    return render(request, 'index1.html')



#def search_wikipedia(request):
    #if request.method == 'POST':
        #search_query = request.POST.get('text1')

        #response = openai.Completion.create(
            #engine="davinci",  # 或者您选择的其他模型
            #prompt=search_query,
            #max_tokens=100  # 根据需求设置最大tokens数
        #)
        #wikipedia_data = response.choices[0].text.strip()

        # 将搜索结果转换为DataFrame
        #data = pd.DataFrame({'search_query': search_query, 'wikipedia_data': wikipedia_data}, index=[0])

        # 将DataFrame转换为CSV格式
        #csv_data = data.to_csv(index=False)

        #response = HttpResponse(content_type='text/csv')
        #response['Content-Disposition'] = 'attachment; filename="wikipedia_search_results.csv"'
        #response.write(csv_data)
        #return response

    #return render(request, 'search.html') # 返回模板渲染结果


openai.api_key = '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'

# download pre-chunked text and pre-computed embeddings
# this file is ~200 MB, so may take a minute depending on your connection speed
embeddings_path = "D:\django\myopenai\myopenai\wikipedia_nag.csv"

df = pd.read_csv(embeddings_path)
# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)

#print(df)

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple([list([str]), list([float])]):
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

# examples
#strings, relatednesses = strings_ranked_by_relatedness("木球金牌", df, top_n=5)
#for string, relatedness in zip(strings, relatednesses):
#    print(f"{relatedness=:.3f}")
#    display(string)

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below articles to answer the subsequent question. If the answer cannot be found in the articles, you do your best to answer questions'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question
    
def ask(request):
    
    model = GPT_MODEL
    token_budget = 4096 - 500
    
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    if request.method == 'POST':
       query = request.POST['text1']
       query = request.POST['text2']
       
    message = query_message(query, df, model=model, token_budget=token_budget)
    
    messages = [
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge" }, #"You answer questions in traditional Chinese about the 中華民國全國大專運動會."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.5
    )
    response_message = response["choices"][0]["message"]["content"]
    dict ={
        'query' : query,
        'response' : response_message
        
        }

    return render(request, 'response.html', dict)



#CATEGORY_TITLE = "Category:中華民國全國大專校院運動會"




def titles_from_category(
    category: mwclient.listing.Category, max_depth: int
) ->set([str]):
    """Return a set of page titles in a given Wiki category and its subcategories."""
    titles = set()
    for cm in category.members():
        if type(cm) == mwclient.page.Page:
            # ^type() used instead of isinstance() to catch match w/ no inheritance
            titles.add(cm.name)
        elif isinstance(cm, mwclient.listing.Category) and max_depth > 0:
            deeper_titles = titles_from_category(cm, max_depth=max_depth - 1)
            titles.update(deeper_titles)
    return titles




# define functions to split Wikipedia pages into sections



def all_subsections_from_section(
    section: mwparserfromhell.wikicode.Wikicode,
    parent_titles: list([str]),
    sections_to_ignore: set([str]),
) -> list([tuple([list([str]), str])]):
    """
    From a Wikipedia section, return a flattened list of all nested subsections.
    Each subsection is a tuple, where:
        - the first element is a list of parent subtitles, starting with the page title
        - the second element is the text of the subsection (but not any children) 
    """
    headings = [str(h) for h in section.filter_headings()]
    title = headings[0]
    if title.strip("=" + " ") in sections_to_ignore:
        # ^wiki headings are wrapped like "== Heading =="
        return []
    titles = parent_titles + [title]
    full_text = str(section)
    section_text = full_text.split(title)[1]
    if len(headings) == 1:
        return [(titles, section_text)]
    else:
        first_subtitle = headings[1]
        section_text = section_text.split(first_subtitle)[0]
        results = [(titles, section_text)]
        for subsection in section.get_sections(levels=[len(titles) + 1]):
            results.extend(all_subsections_from_section(subsection, titles, sections_to_ignore))
        return results

def all_subsections_from_title(
    title: str,
    sections_to_ignore: set([str]) ,
    site_name: str,
) -> list([tuple([list([str]), str])]):  

    """
    From a Wikipedia page title, return a flattened list of all nested subsections.
    Each subsection is a tuple, where:
        - the first element is a list of parent subtitles, starting with the page title
        - the second element is the text of the subsection (but not any children)
    """

    site = mwclient.Site(site_name)
    page = site.pages[title]
    text = page.text()
    parsed_text = mwparserfromhell.parse(text)
    headings = [str(h) for h in parsed_text.filter_headings()]
    if headings:
        summary_text = str(parsed_text).split(headings[0])[0]
    else:
        summary_text = str(parsed_text)
    results = [([title], summary_text)]
    for subsection in parsed_text.get_sections(levels=[2]):
        results.extend(all_subsections_from_section(subsection, [title], sections_to_ignore))
    return results

# split pages into sections
# may take ~1 minute per 100 articles


# clean text
def clean_section(section: tuple([list([str]), str])) -> tuple([list([str]), str]):
    """
    Return a cleaned up section with:
        - <ref>xyz</ref> patterns removed
        - leading/trailing whitespace removed
    """
    titles, text = section
    text = re.sub(r"<ref.*?</ref>", "", text)
    text = text.strip()
    return (titles, text)




# filter out short/blank sections
def keep_section(section: tuple([list([str]), str])) -> bool:
    """Return True if the section should be kept, False otherwise."""
    titles, text = section
    if len(text) < 16:
        return False
    else:
        return True





    


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def halved_by_delimiter(string: str, delimiter: str = "\n") -> list([str, str]):
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # no delimiter found
    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point
    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]

def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string


def split_strings_from_subsection(
    subsection: tuple([list([str]), str]),
   # subsection: str,
    max_tokens: int = 1000,
    model: str = GPT_MODEL,
    max_recursion: int = 5,
) -> list([str]):
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """
    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)
    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
       return [string]
    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
       return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # otherwise, split in half and recurse
    else:
       titles, text = subsection
       for delimiter in ["\n\n", "\n", ". "]:
           left, right = halved_by_delimiter(text, delimiter=delimiter)
           if left == "" or right == "":
            # if either half is empty, retry with a more fine-grained delimiter
              continue
           else:
            # recurse on each half
              results = []
              for half in [left, right]:
                  half_subsection = (titles, half)
                  half_strings = split_strings_from_subsection(
                       half_subsection,
                       max_tokens=max_tokens,
                       model=model,
                       max_recursion=max_recursion - 1,
                )
                  results.extend(half_strings)
                  return results
                  # otherwise no split was found, so just truncate (should be very rare)
                  return [truncated_string(string, model=model, max_tokens=max_tokens)]

# split sections into chunks
def getwikipedia(request):
    WIKI_SITE = "zh.wikipedia.org"
     
    query = request.POST['search_query']
    output = request.POST['output']
    CATEGORY_TITLE = "Category:"
    CATEGORY_TITLE = CATEGORY_TITLE + query
    
    site = mwclient.Site(WIKI_SITE)
    category_page = site.pages[CATEGORY_TITLE]
    titles = titles_from_category(category_page, max_depth=3)
    print(f"Found {len(titles)} article titles in {CATEGORY_TITLE}.")
    
    SECTIONS_TO_IGNORE = [
    "See also",
    "References",
    "External links",
    "Further reading",
    "Footnotes",
    "Bibliography",
    "Sources",
    "Citations",
    "Literature",
    "Footnotes",
    "Notes and references",
    "Photo gallery",
    "Works cited",
    "Photos",
    "Gallery",
    "Notes",
    "References and sources",
    "References and notes",
    ]
     
    wikipedia_sections = []
    for title in titles:
       wikipedia_sections.extend(all_subsections_from_title(title,SECTIONS_TO_IGNORE,WIKI_SITE))
    print(f"Found {len(wikipedia_sections)} sections in {len(titles)} pages.")
    
    
    
    wikipedia_sections = [clean_section(ws) for ws in wikipedia_sections]
    
    original_num_sections = len(wikipedia_sections)
    wikipedia_sections = [ws for ws in wikipedia_sections if keep_section(ws)]
    print(f"Filtered out {original_num_sections-len(wikipedia_sections)} sections, leaving {len(wikipedia_sections)} sections.")
    
    for ws in wikipedia_sections[:5]:
        print(ws[0])
        display(ws[1][:20] + "...")
        print()
    
    
    GPT_MODEL = "gpt-3.5-turbo"  # only matters insofar as it selects which tokenizer to use
    MAX_TOKENS = 1600
    wikipedia_strings = []
    for section in wikipedia_sections:
       # print(section)
        wikipedia_strings.extend(split_strings_from_subsection(section, max_tokens=MAX_TOKENS))

    print(f"{len(wikipedia_sections)} Wikipedia sections split into {len(wikipedia_strings)} strings.")
    
   

    
   
    
    
    original_num_sections = len(wikipedia_sections)
    wikipedia_sections = [ws for ws in wikipedia_sections if keep_section(ws)]
   
    print(f"Filtered out {original_num_sections-len(wikipedia_sections)} sections, leaving {len(wikipedia_sections)} sections.")
    i=0
    while i < 10:
      print("=============")
      print(wikipedia_strings[i])
      i=i+1
    EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
    BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

    embeddings = []
    for batch_start in range(0, len(wikipedia_strings), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = wikipedia_strings[batch_start:batch_end]
        print(f"Batch {batch_start} to {batch_end-1}")
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
    
        for i, be in enumerate(response["data"]):
            assert i == be["index"]  # double check embeddings are in same order as input
            batch_embeddings = [e["embedding"] for e in response["data"]]
            embeddings.extend(batch_embeddings)

    df = pd.DataFrame({"text": wikipedia_strings, "embedding": embeddings})
# save document chunks and embeddings

    SAVE_PATH = output 

    df.to_csv(SAVE_PATH, index=False, encoding='utf-8-sig',mode='w')

    return()





