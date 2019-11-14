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


#%%
#url_base ="https://www.etsy.com/c/home-and-living?ref=pagination&page="
# Drawing and Illustrations
#url_base ="https://www.etsy.com/c/art-and-collectibles/drawing-and-illustration?ref=pagination&explicit=1&page="
# Prints
url_base ="https://www.etsy.com/c/art-and-collectibles/prints?explicit=1&ref=pagination&page="
R = 249
urls=[]
for r in range(R): 
    urls.append(url_base+str(r+2))
    
#urls = ["https://www.etsy.com/c/home-and-living?ref=pagination&page=2",
#        "https://www.etsy.com/c/home-and-living?ref=pagination&page=3",
#        "https://www.etsy.com/c/home-and-living?ref=pagination&page=4",
#        "https://www.etsy.com/c/home-and-living?ref=pagination&page=5"]

page_count = 0

id = []
href = []
price = []
blurb = []
seller = []
stars = []
li_list = []
numrate = []
list_dict=[]

for url in urls:
    page_count+=1
    source = requests.get(url).text
    #source = requests.get("https://www.etsy.com/c/home-and-living?ref=catnav-792").text
    soup = bs(source, "lxml")
    #print(soup.prettify())              
    
    #%%
    #article = soup.find("ul",data-target="category-block-grid", class_="list-unstyled block-grid-xs-2 show-xs show-sm show-md  hide-lg hide-xl hide-tv")
    element = soup.find("ul",{"class":"responsive-listing-grid wt-grid wt-grid--block justify-content-flex-start pl-xs-0"})    
    li_match = element.find_all("li")
    
    #for li in range(150,490):
    li_count=0
    for item in li_match:#[0:20]:   
        li_count+=1
        try: 
            id_i = str(item.find("div",{"class":"js-merch-stash-check-listing"})["data-palette-listing-id"])
            #id.append(id_i)
            
            href_i = str(item.find("a",{"class":"display-inline-block listing-link"})["href"])
            #href.append(href_i)
            
            seller_i = str(item.find("div",{"class":"v2-listing-card__shop"}).p.text)
            #seller.append(seller_i)
            #seller.update({li:seller_i})
            
            # get current price
            price_i = str(item.find("span", {"class":"currency-value"}).text)
            #price.append(price_i)
            #price.update({li:price_i})
        
            # get blurb
            blurb_i = str(item.find("h2",{"class":"text-gray text-truncate mb-xs-0 text-body"}).text) 
            blurb_i = blurb_i.replace("\n","").strip()
            #blurb.update({li:blurb_i})
            
            stars_i = str(item.find("input",{"name":"rating"})["value"]) 
            #stars.append(stars_i)
    
            numrate_i = str(item.find("span",{"class":"text-body-smaller text-gray-lighter display-inline-block icon-b-1"}).text) 
            numrate_i=numrate_i.strip("()").replace(",","")
            #numrate.append(numrate_i)        
            #text-body-smaller text-gray-lighter display-inline-block icon-b-1
            
            row= {'id':id_i,'href':href_i,'seller':seller_i,'price':price_i,'blurb':blurb_i,
                  'stars':stars_i,'numrate':numrate_i}
            list_dict.append(row)
            #row = (id_i,href_i,seller_i,price_i,blurb_i,stars_i,numrate_i)
            #print(row)
            #li_list.append(li)
        except Exception as e:
            break
            #id.append("none")
            #href.append("none")
            #seller.append("none")
            #price.append("none")
            #blurb.append("none")
            #stars.append("none")
            #numrate.append("none")
            #li_list.append(li)
    
#%% Create DataFrame from all the catagory lists
df = pd.DataFrame(list_dict)  
#df.to_csv("etsy_prints.csv")
#df = pd.DataFrame({"li_num":li_list})
# =============================================================================
# df = pd.DataFrame({"id":id})
#      
# #df["li_num"] = li_list
# df["href"] = href
# df["price"] = price
# df["blurb"] = blurb
# df["seller"] = seller
# df["stars"] = stars
# df["numrate"] = numrate
# =============================================================================
