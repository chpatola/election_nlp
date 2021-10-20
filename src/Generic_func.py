"""Generic modelling functions"""
#https://medium.com/@rrfd/cookiecutter-data-science-organize-your-projects-atom-and-jupyter-2be7862f487e

def scandinavian_letters(data):
      data.replace(to_replace="Ã¤", value="ä",regex=True, inplace = True)
      data.replace(value="ö",regex="Ã¶", inplace = True)
      return data

def nyName (data,old,new):
      data.rename(columns={old: new}, inplace = True)
      
