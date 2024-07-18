import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud

#...........................................................................................

def pinta_distribucion_categoricas(df, columnas_categoricas, relativa=False, mostrar_valores=False):
    num_columnas = len(columnas_categoricas)
    num_filas = (num_columnas // 2) + (num_columnas % 2)

    fig, axes = plt.subplots(num_filas, 2, figsize=(15, 5 * num_filas))
    axes = axes.flatten() 

    for i, col in enumerate(columnas_categoricas):
        ax = axes[i]
        if relativa:
            total = df[col].value_counts().sum()
            serie = df[col].value_counts().apply(lambda x: x / total)
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue = serie.index, legend = False)
            ax.set_ylabel('Frecuencia Relativa')
        else:
            serie = df[col].value_counts()
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue = serie.index, legend = False)
            ax.set_ylabel('Frecuencia')

        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

        if mostrar_valores:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    for j in range(i + 1, num_filas * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

#...........................................................................................

def plot_categorical_relationship_fin(df, cat_col1, cat_col2, relative_freq=False, show_values=False, size_group = 5):
    # Prepara los datos
    count_data = df.groupby([cat_col1, cat_col2]).size().reset_index(name='count')
    total_counts = df[cat_col1].value_counts()
    
    # Convierte a frecuencias relativas si se solicita
    if relative_freq:
        count_data['count'] = count_data.apply(lambda x: x['count'] / total_counts[x[cat_col1]], axis=1)

    # Si hay más de size_group categorías en cat_col1, las divide en grupos de size_group
    unique_categories = df[cat_col1].unique()
    if len(unique_categories) > size_group:
        num_plots = int(np.ceil(len(unique_categories) / size_group))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * size_group:(i + 1) * size_group]
            data_subset = count_data[count_data[cat_col1].isin(categories_subset)]

            # Crea el gráfico
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=data_subset, order=categories_subset)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {cat_col1} y {cat_col2} - Grupo {i + 1}')
            plt.xlabel(cat_col1)
            plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, size_group),
                                textcoords='offset points')

            # Muestra el gráfico
            plt.show()
    else:
        # Crea el gráfico para menos de size_group categorías
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=count_data)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {cat_col1} y {cat_col2}')
        plt.xlabel(cat_col1)
        plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, size_group),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.show()

#...........................................................................................

def plot_categorical_numerical_relationship(df, categorical_col, numerical_col, show_values=False, measure='mean'):
    # Calcula la medida de tendencia central (mean o median)
    if measure == 'median':
        grouped_data = df.groupby(categorical_col)[numerical_col].median()
    else:
        # Por defecto, usa la media
        grouped_data = df.groupby(categorical_col)[numerical_col].mean()

    # Ordena los valores
    grouped_data = grouped_data.sort_values(ascending=False)

    # Asegura que siempre haya 20 categorías por gráfico
    unique_categories = grouped_data.index.unique()
    num_plots = int(np.ceil(len(unique_categories) / 20))

    for i in range(num_plots):
        # Selecciona un subconjunto de categorías para cada gráfico
        start_index = i * 20
        end_index = start_index + 20
        categories_subset = unique_categories[start_index:end_index]
        data_subset = grouped_data.loc[categories_subset]

        # Crea el gráfico
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=data_subset.index, y=data_subset.values)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {categorical_col} y {numerical_col} - Grupo {i + 1}')
        plt.xlabel(categorical_col)
        plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.show()

#plot_categorical_numerical_relationship(df, 'categorical_col', 'numerical_col', show_values=True, measure='mean')

#...........................................................................................

def print_unique_value_counts(df, feature):
    # Contar los valores únicos de la columna especificada
    count_by_feature = df[feature].value_counts()
    total_features = count_by_feature.sum()
    percentage_by_feature = (count_by_feature / total_features) * 100

    # Imprimir los resultados en el formato deseado
    print(f"Unique values in '{feature}':")
    print("-----------------------------------------------------")
    print(f"{feature:<20} {'Count':<10} {'%':<10}")
    print("-----------------------------------------------------")
    for value, count, percent in zip(count_by_feature.index, count_by_feature, percentage_by_feature):
        print(f"{value:<20} {count:<10} {percent:.2f}%")

# Ejemplo de uso print_unique_value_counts(df, feature='genre')

#...........................................................................................

def plot_combined_graphs(df, columns, whisker_width=1.5, bins = None):
    num_cols = len(columns)
    if num_cols:
        
        fig, axes = plt.subplots(num_cols, 2, figsize=(12, 5 * num_cols))
        print(axes.shape)

        for i, column in enumerate(columns):
            if df[column].dtype in ['int64', 'float64']:
                # Histograma y KDE
                sns.histplot(df[column], kde=True, ax=axes[i,0] if num_cols > 1 else axes[0], bins= "auto" if not bins else bins)
                if num_cols > 1:
                    axes[i,0].set_title(f'Histograma y KDE de {column}')
                else:
                    axes[0].set_title(f'Histograma y KDE de {column}')

                # Boxplot
                sns.boxplot(x=df[column], ax=axes[i,1] if num_cols > 1 else axes[1], whis=whisker_width)
                if num_cols > 1:
                    axes[i,1].set_title(f'Boxplot de {column}')
                else:
                    axes[1].set_title(f'Boxplot de {column}')

        plt.tight_layout()
        plt.show()

#.................................................

def plot_grouped_boxplots(df, cat_col, num_col):
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)
    group_size = 5

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cat_col, y=num_col, data=subset_df)
        plt.title(f'Boxplots of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xticks(rotation=45)
        plt.show()

#.................................................

def plot_grouped_boxplots_columns_list(df, cat_cols, num_cols):
    if isinstance(cat_cols, str):
        cat_cols = [cat_cols]
    if isinstance(num_cols, str):
        num_cols = [num_cols]
        
    for cat_col in cat_cols:
        unique_cats = df[cat_col].unique()
        for num_col in num_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=cat_col, y=num_col, data=df)
            plt.title(f'Boxplot de {num_col} agrupado por {cat_col}')
            plt.xticks(rotation=45)
            plt.show()

#.................................................

def plot_grouped_histograms(df, cat_col, num_col, group_size):
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        for cat in subset_cats:
            sns.histplot(subset_df[subset_df[cat_col] == cat][num_col], kde=True, label=str(cat))
        
        plt.title(f'Histograms of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xlabel(num_col)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


#.................................................

def plot_top_n_categories(df, cat_cols, n):
    for cat_col in cat_cols:
        # Contar la frecuencia de cada categoría
        cat_counts = df[cat_col].value_counts().nlargest(n)
        
        # Crear un gráfico de barras
        plt.figure(figsize=(10, 6))
        cat_counts.plot(kind='bar', color='skyblue')
        plt.title(f'Top {n} {cat_col}')
        plt.xlabel(cat_col)
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Ejemplo de uso plot_top_n_categories(df, features_categorical, n=25)

#------------------------------------------------

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_wordclouds_for_categorical(df, categorical_cols):
    # Iterar sobre cada columna categórica
    for col in categorical_cols:
        # Unir todos los valores de la columna en una sola cadena
        text = ' '.join(df[col].astype(str).values.tolist())

        # Crear el objeto WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)

        # Mostrar el mapa de palabras
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud de {col}')
        plt.axis('off')
        plt.show()

# Ejemplo de uso plot_wordclouds_for_categorical(df, features_categorical)

#................................................................

def plot_wordclouds_for_categorical_by_group(df, group_column, categorical_col):
    # Agrupar el DataFrame por la columna especificada
    grouped_df = df.groupby(group_column)
    
    # Iterar sobre cada grupo
    for group_value, group_data in grouped_df:
        # Unir todos los valores de la columna en una sola cadena para el grupo actual
        text = ' '.join(group_data[categorical_col].astype(str).values.tolist())

        # Crear el objeto WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)

        # Mostrar el mapa de palabras
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud of {group_value} in {group_column}')
        plt.axis('off')
        plt.show()

# Ejemplo de uso plot_wordclouds_for_categorical_by_group(df, group_column='genre', categorical_col='artist_name')

#···················································································································

# Función para crear la tabla de outliers por para cada feature
def outliers_by_feature_table(df):
    # Crear un diccionario para almacenar los resultados por feature
    results = {}
    
    # Obtener la lista de features (columnas numéricas)
    features = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Iterar sobre cada feature
    for feature in features:
        # Calcular outliers por género para el feature actual
        outliers_by_genre = df.groupby('genre')[feature].apply(lambda x: (x > 1.5 * x.quantile(0.75)) | (x < 1.5 * x.quantile(0.25))).sum()
        
        # Ordenar de mayor a menor frecuencia de outliers
        outliers_by_genre = outliers_by_genre.sort_values(ascending=False)
        
        # Almacenar los resultados en el diccionario
        results[feature] = outliers_by_genre
    
    return results

#···················································································································

def value_table_filtered(df, filter_column, filter_type, filter_value1, filter_value2=None):
    # Filtrar el DataFrame según el tipo de filtro
    if filter_type == 'greater_than':
        df_filtered = df[df[filter_column] > filter_value1]
    elif filter_type == 'less_than':
        df_filtered = df[df[filter_column] < filter_value1]
    elif filter_type == 'between':
        if filter_value2 is None:
            raise ValueError("Para el filtro 'between', se deben proporcionar dos valores de filtro.")
        df_filtered = df[(df[filter_column] > filter_value1) & (df[filter_column] < filter_value2)]
    else:
        raise ValueError("Tipo de filtro no soportado. Use 'greater_than', 'less_than' o 'between'.")
    
    # Crear un diccionario para almacenar los resultados por feature
    results = {}
    
    # Obtener la lista de features (columnas numéricas excluyendo la columna de filtro)
    numeric_columns = df_filtered.select_dtypes(include=['int64', 'float64']).columns.drop([filter_column])
    
    # Iterar sobre cada feature
    for feature in numeric_columns:
        # Calcular el conteo de cada valor en el feature
        value_counts = df_filtered[feature].value_counts().reset_index()
        value_counts.columns = [feature, 'Count']
        
        # Calcular el porcentaje
        value_counts['%'] = (value_counts['Count'] / value_counts['Count'].sum()) * 100
        
        # Ordenar la tabla por el valor de conteo más alto
        value_counts = value_counts.sort_values(by='Count', ascending=False)
        
        # Preparar el encabezado dinámico
        if filter_type == 'between':
            filter_description = f"{filter_column} {filter_type.replace('_', ' ')} {filter_value1} y {filter_value2}"
        else:
            filter_description = f"{filter_column} {filter_type.replace('_', ' ')} {filter_value1}"
        
        # Imprimir el encabezado de la tabla
        print(f"{feature} - {filter_description}:")
        print("-----------------------------------------------------")
        print(f"{feature:<20} {'Count':<10} {'%':<10}")
        print("-----------------------------------------------------")
        
        # Imprimir cada fila de la tabla
        for _, row in value_counts.iterrows():
            print(f"{row[feature]:<20} {row['Count']:<10} {row['%']:<10.2f}")
        
        # Imprimir una línea separadora entre tablas
        print("\n")
        
        # Almacenar los resultados en el diccionario
        results[feature] = value_counts
    
    return results

#Ejemplos de uso: value_table_filtered(df, 'popularity', 'greater_than', 50)
#value_table_filtered(df, 'popularity', 'less_than', 20)
#value_table_filtered(df, 'popularity', 'between', 20, 50)


#.....................................................................................................................
def grafico_dispersion_con_correlacion(df, columna_x, columna_y, tamano_puntos=50, mostrar_correlacion=False):
    """
    Crea un diagrama de dispersión entre dos columnas y opcionalmente muestra la correlación.

    Args:
    df (pandas.DataFrame): DataFrame que contiene los datos.
    columna_x (str): Nombre de la columna para el eje X.
    columna_y (str): Nombre de la columna para el eje Y.
    tamano_puntos (int, opcional): Tamaño de los puntos en el gráfico. Por defecto es 50.
    mostrar_correlacion (bool, opcional): Si es True, muestra la correlación en el gráfico. Por defecto es False.
    """

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=columna_x, y=columna_y, s=tamano_puntos)

    if mostrar_correlacion:
        correlacion = df[[columna_x, columna_y]].corr().iloc[0, 1]
        plt.title(f'Diagrama de Dispersión con Correlación: {correlacion:.2f}')
    else:
        plt.title('Diagrama de Dispersión')

    plt.xlabel(columna_x)
    plt.ylabel(columna_y)
    plt.grid(True)
    plt.show()


def bubble_plot(df, col_x, col_y, col_size, scale = 1000):
    """
    Crea un scatter plot usando dos columnas para los ejes X e Y,
    y una tercera columna para determinar el tamaño de los puntos.

    Args:
    df (pd.DataFrame): DataFrame de pandas.
    col_x (str): Nombre de la columna para el eje X.
    col_y (str): Nombre de la columna para el eje Y.
    col_size (str): Nombre de la columna para determinar el tamaño de los puntos.
    """

    # Asegúrate de que los valores de tamaño sean positivos
    sizes = (df[col_size] - df[col_size].min() + 1)/scale

    plt.scatter(df[col_x], df[col_y], s=sizes)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f'Burbujas de {col_x} vs {col_y} con Tamaño basado en {col_size}')
    plt.show()


#.......................................................................................................

# MODELOS

#.......................................................................................................

def show_coefs(model, figsize=(10, 5)):
    df_coef = pd.DataFrame(model.coef_, index=model.feature_names_in_, columns=["coefs"])

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    df_coef.plot(kind="barh", ax=ax[0], legend=False)
    df_coef.abs().sort_values(by="coefs").plot(kind="barh", ax=ax[1], legend=False)
    fig.suptitle("Model Coefficients")

    fig.tight_layout()

    return df_coef

