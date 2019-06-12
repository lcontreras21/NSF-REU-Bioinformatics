import bs4
import requests
import time 
import pickle

from nn_partial_links import gene_dict


partial_link = "https://www.genecards.org/cgi-bin/carddisp.pl?gene="

g = open("text_files/missing_genes.txt", "r")
gene_list = g.readline().split()
HEADER = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'}

def get_gene(string):
	gene_index = string.index("gene=")
	end_index = string.index('");')
	return string[gene_index + 5: end_index]

def google(gene):
	res = requests.get("https://search.yahoo.com/search?ei=UTF-8&fr=crmas&p=" + gene + "+genecards", headers=HEADER)
	res.raise_for_status()
	soup = bs4.BeautifulSoup(res.text, "html.parser")
	link_elems = soup.select('.title a')
	# the above opens up a yahoo search page and gets the first 
	# link which is 99 percent likely to be the one I want
	res = requests.get(link_elems[0].get('href'), allow_redirects=True)
	soup = bs4.BeautifulSoup(res.text, "html.parser")
	try:
		new_name = get_gene(str(soup))
	except:
		print("Check this one manually:", gene)
		new_name = "Check manually"
		pass
	print(new_name)
	gene_names[gene] = new_name

if __name__ == "__main__":
	gene_names = {}
	for gene in gene_list:
		google(gene)
		#time.sleep(0.5)
	gene_names["SLMO2"] = "PRELID3B"
	gene_names["SEPW1"] = "SELENOW"
	gene_names["TARP"] = "CD3G"
	gene_names["CRIPAK"] = "UVSSA"
	gene_names["FTSJD2"] = "CMTR1"
	gene_names["B1P1"] = "XBP1P1"
	from nn_partial_links import gene_dict
	genes = gene_dict()
	print("Genes that are still not accounted for.")
	for gene in gene_names:
		try:
			genes[gene_names[gene]]
		except:
			print(gene)
	
	print("Saving gene name pairs to file")
	gene_file = open("text_files/gene_pairs.pickle", "wb")
	pickle.dump(gene_names, gene_file)
	gene_file.close()
