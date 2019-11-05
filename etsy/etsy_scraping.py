# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:49:40 2019

@author: ewell
"""
# Corey Schafer
# https://www.youtube.com/watch?v=ng2o98k983k

from bs4 import BeautifulSoup as bs
import requests
import pandas as pd

# source = file oath
#with open(source) as html_file:
#with open("http://www.ohiostatehouse.org/") as html_file:
#    soup = bs(html_file, "lxml")
#    print("hello")

#%%
source = requests.get("https://www.etsy.com/c/home-and-living?ref=catnav-891").text
soup = bs(source, "lxml")
print(soup.prettify())              
#%%

#article = soup.find("ul",data-target="category-block-grid", class_="list-unstyled block-grid-xs-2 show-xs show-sm show-md  hide-lg hide-xl hide-tv")
element = soup.find("ul")
#print(element["role"])
#print(element.contents)

for child in element.children:
    print("CHILD:"+str(child))

#print(element.prettify())
list_id=464949133
#data-listing-id="464949133"


#print(soup.prettify())
#print(soup.title)
#print(soup.text)

#%%
li_match = soup.find_all("li")

price = {}
link = {}
seller = {}
li_list = []

for li in range(450,490):
    
    matchi=li_match[li]
    #tagi = matchi.find("span", class_="currency-value")
    
    seller_li = str(matchi.find("class":"v2-listing-card__shop"}).text)
    
    # get current price
    price_li = str(matchi.find("span", {"class":"currency-value"}).text)
    price.update({li:price_li})

    link_li = str(matchi.find("h2",{"class":"text-gray text-truncate mb-xs-0 text-body"}).text)   
    link.update({li:link_li})
    
    li_list.append(li)
    
    #print(linki.text)
#%%
df = pd.DataFrame()
     
df["price"] = price
df["link"] = link
