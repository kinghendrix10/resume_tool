#!/usr/bin/env python
# coding: utf-8

# In[635]:


import PyPDF2
import os
import re
import sys
from spacy.matcher import PhraseMatcher
from collections import Counter
import en_core_web_sm
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from spacy.matcher import Matcher


# In[648]:


# # mypath = 'C:/Users/Huleji/Documents/CV and Resume/Sample/' #enter your path here where you saved the resumes
# mypath = input('Enter folder path: ')
# assert os.path.exists(mypath), 'Files not found at '+str(mypath)
# # mypath = open(user_input, 'r+')
# onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
# resume_tool(onlyfiles)[0]


# In[637]:


def extract_func(file):
    fileReader = PyPDF2.PdfFileReader(open(file,'rb'))
    countpage = fileReader.getNumPages()
    count = 0
    resume = []
    while count < countpage:    
        pageObj = fileReader.getPage(count)
        count +=1
        txt = pageObj.extractText()
        resume.append(txt)
    return resume


# In[638]:


def candidate_table(file, cand_df=pd.DataFrame(), cand_profile=pd.DataFrame(), nlp = en_core_web_sm.load()):
    
    resume = extract_func(file)
    res_text = str(resume)
    res_text = res_text.replace("\\n", "")
    res_text = res_text.lower()
    
    matcher = Matcher(nlp.vocab)
    nlp_text = nlp(res_text)
    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    
    matcher.add('FULL NAME', None, pattern)
    
    matches = matcher(nlp_text)
    
    match_id, start, end = matches[0]
    full_name = nlp_text[start:end]
    full_name
    
    cand_full_name = pd.read_csv(StringIO(full_name.text), names = ['Candidate Name'])
#     for match_id, start, end in matches:
#         full_name = nlp_text[start:end]
#         print (full_name.text)

    # reg = re.compile(r'.*?(\(?\d{6}\D{0,3}\d{3}\D{0,3}\d{4}\D{0,3}).*?', re.S)
    reg = re.compile(r'.*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}\D{0,3}).*?', re.S)

    phone = re.findall(reg, res_text)
    numbers = []
    for i in range (0, len(phone)):
        if phone[i]:
            number = ''.join(phone[i])
            number = re.sub(r'[a-z]|[\s\-\.\+]','', number)
            if len(number) > 11:
                number = '+' + number
            else:
                number = '+234' + number
            numbers.append(number)
#     return numbers
    email = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', res_text)
    
    education = []
    dip_deg = re.findall(r'(o|ordinary)?[-\s\.]?(a|advanced)?[-\s\.]+(n|national)?[-\s\.]?(diploma|dip|d)[-\s\.]+', res_text)
    # dip_deg = re.findall(r'(\w)[-\s\.]?(d|diploma)[-\s\.]?', res_text)
    for i in range (0, len(dip_deg)):
        if dip_deg[i]:
            dip_deg = ' '.join(dip_deg[i])
        
# bachelor_deg = re.findall(r'(b|bachelor)[-\s\.]+(a|arts|sc|science|ed|edu|education|eng|engineering|tech|technology)[-\s\.]+', res_text) 
# bachelor_deg = re.findall(r'(b|bachelor)[-\s\.]+(of)*[-\s\.]+(a|arts|sc|science|ed|edu|education|eng|engineering|tech|technology)[-\s\.]+', res_text) 
    bachelor_deg = re.findall(r'(b|bachelor)[-\s\.]?(of)*[-\s\.]?(a|arts|sc|science|ed|edu|education|eng|engineering|tech|technology)[-\s\.\,]+', res_text) 
    for i in range (0, len(bachelor_deg)):
        if bachelor_deg[i]:
            bachelor_deg = ' '.join(bachelor_deg[i])
        
    pgd_deg = re.findall(r'(p|post)[-\s\.]+(g|graduate)[-\s\.]+(d|diploma)[-\s\.]+', res_text)
    for i in range (0, len(pgd_deg)):
        if pgd_deg[i]:
            pgd_deg = ' '.join(pgd_deg[i])
        
    masters_deg = re.findall(r'(m|masters?)[-\s\.]+(of)*[-\s\.]+(a|arts|sc|science|ed|edu|education|eng|engineering|tech|technology|b|business)[-\s\.]+', res_text)
    for i in range (0, len(masters_deg)):
        if masters_deg[i]:
            masters_deg = ' '.join(masters_deg[i])
        
    phd_deg = re.findall(r'(p|doctor)[-\s\.]+(h)?[-\s\.]+(d|philosophy)[-\s\.]+', res_text)
    for i in range (0, len(phd_deg)):
        if phd_deg[i]:
            phd_deg = ' '.join(phd_deg[i])
        
    if len(dip_deg) != 0:
        education.append(dip_deg)
    if len(bachelor_deg) != 0:
        education.append(bachelor_deg)
    if len(pgd_deg) != 0:
        education.append(pgd_deg)
    if len(masters_deg) != 0:
        education.append(masters_deg)
    if len(phd_deg) != 0:
        education.append(phd_deg)

#     return education
    
    text = res_text
    #below is the csv where we have all the keywords, you can customize your own
    keyword_dict = pd.read_csv('skills.csv', encoding='ISO-8859-1')
#     Stats_words = [nlp(text) for text in keyword_dict['Statistics'].dropna(axis = 0)]
#     NLP_words = [nlp(text) for text in keyword_dict['NLP'].dropna(axis = 0)]
#     ML_words = [nlp(text) for text in keyword_dict['Machine Learning'].dropna(axis = 0)]
#     DL_words = [nlp(text) for text in keyword_dict['Deep Learning'].dropna(axis = 0)]
#     R_words = [nlp(text) for text in keyword_dict['R Language'].dropna(axis = 0)]
#     Python_words = [nlp(text) for text in keyword_dict['Python Language'].dropna(axis = 0)]
#     Data_Engineering_words = [nlp(text) for text in keyword_dict['Data Engineering'].dropna(axis = 0)]

    doc = nlp(text)
    d = []
    x = []
    
    for i in range(0, keyword_dict.shape[1]):
        skill_words = [nlp(text) for text in keyword_dict[keyword_dict.columns[i]].dropna(axis = 0)]
        matcher = PhraseMatcher(nlp.vocab)
        matcher.add(keyword_dict.columns[i], None, *skill_words)
    
        matches = matcher(doc)
        for match_id, start, end in matches:
            rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
            span = doc[start : end]  # get the matched slice of the doc
            d.append((rule_id, span.text))
            x.append(span.text)
    
        keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())

    df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
    df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
    df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))

    base = os.path.basename(file)
    filename = os.path.splitext(base)[0]
       
    name = filename.split('_')
    name = name[0].lower()
    # converting str to dataframe
    doc_name = pd.read_csv(StringIO(name), names = ['File Name'])


    dataframe = pd.concat([doc_name['File Name'], cand_full_name['Candidate Name'],
                           df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
    dataframe['File Name'] = dataframe['File Name'].fillna(doc_name['File Name'].iloc[0])
    dataframe['Candidate Name'] = dataframe['Candidate Name'].fillna(cand_full_name['Candidate Name'].iloc[0])

    cand_df = pd.concat([cand_df, dataframe], axis=0)

    profile = pd.DataFrame()
    

    profile['Name'] = str(full_name).strip('[]'),
    profile['Contact'] = ','.join(numbers),
    profile['Email'] = ''.join(email),
    profile['Education'] = ','.join(education),
    profile['Skills'] = ','.join(x)

    cand_profile = pd.concat([cand_profile, profile], axis=0)
    
    return cand_profile, cand_df

def cand_graph(dataframe):
    final_df = dataframe['Keyword'].groupby([dataframe['Candidate Name'], dataframe['Subject']]).count().unstack()
    final_df.reset_index(inplace = True)
    final_df.fillna(0,inplace=True)
    new_df = final_df.iloc[:,1:]
    new_df.index = final_df['Candidate Name']

    plt.rcParams.update({'font.size': 10})
    ax = new_df.plot.barh(title="Resume keywords by category", legend=False, figsize=(25,7), stacked=True)
    labels = []
    for j in new_df.columns:
        for i in new_df.index:
            label = str(j)+": " + str(new_df.loc[i][j])
            labels.append(label)
    patches = ax.patches
    for label, rect in zip(labels, patches):
        width = rect.get_width()
        if width > 0:
            x = rect.get_x()
            y = rect.get_y()
            height = rect.get_height()
            ax.text(x + width/2., y + height/2., label, ha='center', va='center')
    return plt


# In[644]:


def resume_tool(onlyfiles, database = pd.DataFrame(),     database1 = pd.DataFrame()):  
    i = 0 
    while i < len(onlyfiles):
        file = onlyfiles[i]
        dat, dat1 = candidate_table(file)
        database, database1 = database.append(dat), database1.append(dat1)
        i +=1
    return database, cand_graph(database1)


# In[640]:


# database


# In[641]:


# database1


# In[642]:


# cand_graph(database1)


# In[546]:


# keyword_dict = pd.read_csv('skills.csv', encoding='ISO-8859-1')

# Stats_words = [nlp(text) for text in keyword_dict['Statistics'].dropna(axis = 0)]
# NLP_words = [nlp(text) for text in keyword_dict['NLP'].dropna(axis = 0)]
# ML_words = [nlp(text) for text in keyword_dict['Machine Learning'].dropna(axis = 0)]
# DL_words = [nlp(text) for text in keyword_dict['Deep Learning'].dropna(axis = 0)]
# R_words = [nlp(text) for text in keyword_dict['R Language'].dropna(axis = 0)]
# Python_words = [nlp(text) for text in keyword_dict['Python Language'].dropna(axis = 0)]
# Data_Engineering_words = [nlp(text) for text in keyword_dict['Data Engineering'].dropna(axis = 0)]


# In[558]:


# keyword_dict.columns[0]


# In[559]:


# Stats_words


# In[571]:


# df


# In[548]:


# keyword_dict


# In[549]:


# from spacy.matcher import PhraseMatcher
# from collections import Counter


# nlp = en_core_web_sm.load()

# matcher = PhraseMatcher(nlp.vocab)
# matcher.add('Stats', None, *Stats_words)
# matcher.add('NLP', None, *NLP_words)
# matcher.add('ML', None, *ML_words)
# matcher.add('DL', None, *DL_words)
# matcher.add('R', None, *R_words)
# matcher.add('Python', None, *Python_words)
# matcher.add('DE', None, *Data_Engineering_words)
# doc = nlp(text)
    
# d = []
# x = []
# matches = matcher(doc)
# for match_id, start, end in matches:
#     rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
#     span = doc[start : end]  # get the matched slice of the doc
#     d.append((rule_id, span.text))
#     x.append(span.text)
# keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())
    
# ## convertimg string of keywords to dataframe
# df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
# df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
# df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
# df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
# df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
    
# #     base = os.path.basename(file)
# #     filename = os.path.splitext(base)[0]
       
# #     name = filename.split('_')
# #     name2 = name[0]
# #     name2 = name2.lower()
# #     ## converting str to dataframe
# #     name3 = pd.read_csv(StringIO(name2),names = ['Candidate Name'])

# # dataframe = pd.DataFrame()
# # dataframe['Candidate Name'] = cand_name
# # dataframe = pd.concat([dataframe, df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
# # dataframe['Candidate Name'] = dataframe['Candidate Name'].fillna(cand_name, inplace = True)

# #     return(dataf)


# In[550]:


# import pandas as pd
# from io import StringIO

# base = os.path.basename(file)
# filename = os.path.splitext(base)[0]
       
# name = filename.split('_')
# name = name[0].lower()
# # converting str to dataframe
# doc_name = pd.read_csv(StringIO(name), names = ['Candidate Name'])


# # match_id, start, end = matches[0]
# span = nlp_text[start:end]
# print (end_name)
# cand_name

# dataframe = pd.DataFrame()
# dataframe['Candidate Name'] = str(cand_name)
# dataframe = pd.concat([doc_name['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
# dataframe['Candidate Name'] = dataframe['Candidate Name'].fillna(doc_name['Candidate Name'].iloc[0])

# dataframe


# In[551]:


# profile = pd.DataFrame()
# profile['Name'] = str(name).strip('[]'),
# profile['Contact'] = ','.join(numbers),
# profile['Email'] = ''.join(email),
# profile['Education'] = ','.join(education),
# profile['Skills'] = ','.join(x)


# In[552]:


# profile


# In[172]:


# final_df = dataframe['Keyword'].groupby([dataframe['Candidate Name'], dataframe['Subject']]).count().unstack()
# final_df.reset_index(inplace = True)
# final_df.fillna(0,inplace=True)
# new_df = final_df.iloc[:,1:]
# new_df.index = final_df['Candidate Name']


# In[174]:


# new_df


# In[175]:


# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 10})
# ax = new_df.plot.barh(title="Resume keywords by category", legend=False, figsize=(25,7), stacked=True)
# labels = []
# for j in new_df.columns:
#     for i in new_df.index:
#         label = str(j)+": " + str(new_df.loc[i][j])
#         labels.append(label)
# patches = ax.patches
# for label, rect in zip(labels, patches):
#     width = rect.get_width()
#     if width > 0:
#         x = rect.get_x()
#         y = rect.get_y()
#         height = rect.get_height()
#         ax.text(x + width/2., y + height/2., label, ha='center', va='center')
# plt.show()

