def analyse_comp_dataset(dataset):

    if dataset.shape[0] <= 10:
                taille_sample = dataset.shape[0]
                
    else:
        taille_sample = 10

    print('\nCe dataset est de taille :', dataset.shape, '\n')

    print('{:>45} {:>15}  {:>15} {:>15} %'.format(
          "variable name", "data not NaN","type", "Filling Rate"))

    for i in dataset.columns:
        a = dataset[i].notnull().sum()
        b = '{:.2f}'.format(a/dataset.shape[0]*100)
        c = str(dataset[i].dtype)

        print('{:>45} {:>15}  {:>15} {:>15}  %'.format(i,a, c, b))


    for j in dataset.columns:
        print('\nInformations sur la colonne ', j)
        print('echantillon aleatoire des valeurs de ', j)
        print(dataset[j].sample(n=taille_sample))
        print('\nstatistiques :')
        print(dataset[j].describe(),'\n')

def filling_rate(dataset):

    print('{:>15} {:>45} {:>15}  {:>15} {:>15}  %'.format(
        "no","variable name", "data not NaN","type", "Filling Rate"))

    for i in range(0,len(dataset.columns)):
        a = dataset[dataset.columns[i]].notnull().sum()
        b = '{:.4f}'.format(a/dataset.shape[0]*100)
        c = str(dataset[dataset.columns[i]].dtype)
        print('{:>15} {:>45} {:>15}  {:>15} {:>15} %'.format(i,dataset.columns[i],a, c, b))

def analyse (dataset):
    rep=''
    while rep not in ['o','n']:
        rep = input("Voulez vous une analyse complète o/n ?")

    if rep == 'o':
        analyse_comp_dataset(dataset)
    else:
        filling_rate(dataset)

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
	

def val_tri(val, decroissant=True):
    dico = {}
    for i in val:
        previous_count = dico.get(i, 0)
        dico[i] = previous_count + 1
    return dict(sorted(dico.items(), key=lambda t: t[1], reverse=decroissant))
