{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "2177a7c6",
   "metadata": {},
   "source": [
    "id: unique id for a news article\n",
    "    \n",
    "title: the title of a news article\n",
    "    \n",
    "author: author of the news article\n",
    "    \n",
    "text: the text of the article; could be incomplete\n",
    "    \n",
    "label: a label that marks whether the news article is real or fake\n",
    "    \n",
    "0.Fake news\n",
    "\n",
    "1.real news"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "id": "eea51af3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import re\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem.porter import PorterStemmer\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "id": "4351a52a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to C:\\Users\\Win\n",
      "[nltk_data]     10\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 166,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import nltk\n",
    "nltk.download('stopwords')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "id": "0856dd52",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', \"you're\", \"you've\", \"you'll\", \"you'd\", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', \"she's\", 'her', 'hers', 'herself', 'it', \"it's\", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', \"that'll\", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', \"don't\", 'should', \"should've\", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', \"aren't\", 'couldn', \"couldn't\", 'didn', \"didn't\", 'doesn', \"doesn't\", 'hadn', \"hadn't\", 'hasn', \"hasn't\", 'haven', \"haven't\", 'isn', \"isn't\", 'ma', 'mightn', \"mightn't\", 'mustn', \"mustn't\", 'needn', \"needn't\", 'shan', \"shan't\", 'shouldn', \"shouldn't\", 'wasn', \"wasn't\", 'weren', \"weren't\", 'won', \"won't\", 'wouldn', \"wouldn't\"]\n"
     ]
    }
   ],
   "source": [
    "# printing stopwords in   English\n",
    "print (stopwords.words('english'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "id": "cb880ab4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# loding the dataset to pandas Data Fream\n",
    "news_dataset = pd.read_csv('E:/contentrain.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "id": "fc087e12",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(20800, 5)"
      ]
     },
     "execution_count": 169,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "news_dataset.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "id": "ed8f517a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>title</th>\n",
       "      <th>author</th>\n",
       "      <th>text</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>\n",
       "      <td>Darrell Lucus</td>\n",
       "      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>FLYNN: Hillary Clinton, Big Woman on Campus - ...</td>\n",
       "      <td>Daniel J. Flynn</td>\n",
       "      <td>Ever get the feeling your life circles the rou...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>Why the Truth Might Get You Fired</td>\n",
       "      <td>Consortiumnews.com</td>\n",
       "      <td>Why the Truth Might Get You Fired October 29, ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>15 Civilians Killed In Single US Airstrike Hav...</td>\n",
       "      <td>Jessica Purkiss</td>\n",
       "      <td>Videos 15 Civilians Killed In Single US Airstr...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>Iranian woman jailed for fictional unpublished...</td>\n",
       "      <td>Howard Portnoy</td>\n",
       "      <td>Print \\nAn Iranian woman has been sentenced to...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id                                              title              author  \\\n",
       "0   0  House Dem Aide: We Didn’t Even See Comey’s Let...       Darrell Lucus   \n",
       "1   1  FLYNN: Hillary Clinton, Big Woman on Campus - ...     Daniel J. Flynn   \n",
       "2   2                  Why the Truth Might Get You Fired  Consortiumnews.com   \n",
       "3   3  15 Civilians Killed In Single US Airstrike Hav...     Jessica Purkiss   \n",
       "4   4  Iranian woman jailed for fictional unpublished...      Howard Portnoy   \n",
       "\n",
       "                                                text  label  \n",
       "0  House Dem Aide: We Didn’t Even See Comey’s Let...      1  \n",
       "1  Ever get the feeling your life circles the rou...      0  \n",
       "2  Why the Truth Might Get You Fired October 29, ...      1  \n",
       "3  Videos 15 Civilians Killed In Single US Airstr...      1  \n",
       "4  Print \\nAn Iranian woman has been sentenced to...      1  "
      ]
     },
     "execution_count": 170,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# print the first 5 of the dataframe\n",
    "news_dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "id": "9e99e5ae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "id           0\n",
       "title      558\n",
       "author    1957\n",
       "text        39\n",
       "label        0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 171,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# councting the number of missing in the data set\n",
    "news_dataset.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "id": "44b497f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# merging the author and news title\n",
    "news_dataset['content'] = news_dataset['author'] +' '+news_dataset['title']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "id": "2b1743cd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>title</th>\n",
       "      <th>author</th>\n",
       "      <th>text</th>\n",
       "      <th>label</th>\n",
       "      <th>content</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>\n",
       "      <td>Darrell Lucus</td>\n",
       "      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>\n",
       "      <td>1</td>\n",
       "      <td>Darrell Lucus House Dem Aide: We Didn’t Even S...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>FLYNN: Hillary Clinton, Big Woman on Campus - ...</td>\n",
       "      <td>Daniel J. Flynn</td>\n",
       "      <td>Ever get the feeling your life circles the rou...</td>\n",
       "      <td>0</td>\n",
       "      <td>Daniel J. Flynn FLYNN: Hillary Clinton, Big Wo...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>Why the Truth Might Get You Fired</td>\n",
       "      <td>Consortiumnews.com</td>\n",
       "      <td>Why the Truth Might Get You Fired October 29, ...</td>\n",
       "      <td>1</td>\n",
       "      <td>Consortiumnews.com Why the Truth Might Get You...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>15 Civilians Killed In Single US Airstrike Hav...</td>\n",
       "      <td>Jessica Purkiss</td>\n",
       "      <td>Videos 15 Civilians Killed In Single US Airstr...</td>\n",
       "      <td>1</td>\n",
       "      <td>Jessica Purkiss 15 Civilians Killed In Single ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>Iranian woman jailed for fictional unpublished...</td>\n",
       "      <td>Howard Portnoy</td>\n",
       "      <td>Print \\nAn Iranian woman has been sentenced to...</td>\n",
       "      <td>1</td>\n",
       "      <td>Howard Portnoy Iranian woman jailed for fictio...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20795</th>\n",
       "      <td>20795</td>\n",
       "      <td>Rapper T.I.: Trump a ’Poster Child For White S...</td>\n",
       "      <td>Jerome Hudson</td>\n",
       "      <td>Rapper T. I. unloaded on black celebrities who...</td>\n",
       "      <td>0</td>\n",
       "      <td>Jerome Hudson Rapper T.I.: Trump a ’Poster Chi...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20796</th>\n",
       "      <td>20796</td>\n",
       "      <td>N.F.L. Playoffs: Schedule, Matchups and Odds -...</td>\n",
       "      <td>Benjamin Hoffman</td>\n",
       "      <td>When the Green Bay Packers lost to the Washing...</td>\n",
       "      <td>0</td>\n",
       "      <td>Benjamin Hoffman N.F.L. Playoffs: Schedule, Ma...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20797</th>\n",
       "      <td>20797</td>\n",
       "      <td>Macy’s Is Said to Receive Takeover Approach by...</td>\n",
       "      <td>Michael J. de la Merced and Rachel Abrams</td>\n",
       "      <td>The Macy’s of today grew from the union of sev...</td>\n",
       "      <td>0</td>\n",
       "      <td>Michael J. de la Merced and Rachel Abrams Macy...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20798</th>\n",
       "      <td>20798</td>\n",
       "      <td>NATO, Russia To Hold Parallel Exercises In Bal...</td>\n",
       "      <td>Alex Ansary</td>\n",
       "      <td>NATO, Russia To Hold Parallel Exercises In Bal...</td>\n",
       "      <td>1</td>\n",
       "      <td>Alex Ansary NATO, Russia To Hold Parallel Exer...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20799</th>\n",
       "      <td>20799</td>\n",
       "      <td>What Keeps the F-35 Alive</td>\n",
       "      <td>David Swanson</td>\n",
       "      <td>David Swanson is an author, activist, journa...</td>\n",
       "      <td>1</td>\n",
       "      <td>David Swanson What Keeps the F-35 Alive</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>20800 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          id                                              title  \\\n",
       "0          0  House Dem Aide: We Didn’t Even See Comey’s Let...   \n",
       "1          1  FLYNN: Hillary Clinton, Big Woman on Campus - ...   \n",
       "2          2                  Why the Truth Might Get You Fired   \n",
       "3          3  15 Civilians Killed In Single US Airstrike Hav...   \n",
       "4          4  Iranian woman jailed for fictional unpublished...   \n",
       "...      ...                                                ...   \n",
       "20795  20795  Rapper T.I.: Trump a ’Poster Child For White S...   \n",
       "20796  20796  N.F.L. Playoffs: Schedule, Matchups and Odds -...   \n",
       "20797  20797  Macy’s Is Said to Receive Takeover Approach by...   \n",
       "20798  20798  NATO, Russia To Hold Parallel Exercises In Bal...   \n",
       "20799  20799                          What Keeps the F-35 Alive   \n",
       "\n",
       "                                          author  \\\n",
       "0                                  Darrell Lucus   \n",
       "1                                Daniel J. Flynn   \n",
       "2                             Consortiumnews.com   \n",
       "3                                Jessica Purkiss   \n",
       "4                                 Howard Portnoy   \n",
       "...                                          ...   \n",
       "20795                              Jerome Hudson   \n",
       "20796                           Benjamin Hoffman   \n",
       "20797  Michael J. de la Merced and Rachel Abrams   \n",
       "20798                                Alex Ansary   \n",
       "20799                              David Swanson   \n",
       "\n",
       "                                                    text  label  \\\n",
       "0      House Dem Aide: We Didn’t Even See Comey’s Let...      1   \n",
       "1      Ever get the feeling your life circles the rou...      0   \n",
       "2      Why the Truth Might Get You Fired October 29, ...      1   \n",
       "3      Videos 15 Civilians Killed In Single US Airstr...      1   \n",
       "4      Print \\nAn Iranian woman has been sentenced to...      1   \n",
       "...                                                  ...    ...   \n",
       "20795  Rapper T. I. unloaded on black celebrities who...      0   \n",
       "20796  When the Green Bay Packers lost to the Washing...      0   \n",
       "20797  The Macy’s of today grew from the union of sev...      0   \n",
       "20798  NATO, Russia To Hold Parallel Exercises In Bal...      1   \n",
       "20799    David Swanson is an author, activist, journa...      1   \n",
       "\n",
       "                                                 content  \n",
       "0      Darrell Lucus House Dem Aide: We Didn’t Even S...  \n",
       "1      Daniel J. Flynn FLYNN: Hillary Clinton, Big Wo...  \n",
       "2      Consortiumnews.com Why the Truth Might Get You...  \n",
       "3      Jessica Purkiss 15 Civilians Killed In Single ...  \n",
       "4      Howard Portnoy Iranian woman jailed for fictio...  \n",
       "...                                                  ...  \n",
       "20795  Jerome Hudson Rapper T.I.: Trump a ’Poster Chi...  \n",
       "20796  Benjamin Hoffman N.F.L. Playoffs: Schedule, Ma...  \n",
       "20797  Michael J. de la Merced and Rachel Abrams Macy...  \n",
       "20798  Alex Ansary NATO, Russia To Hold Parallel Exer...  \n",
       "20799            David Swanson What Keeps the F-35 Alive  \n",
       "\n",
       "[20800 rows x 6 columns]"
      ]
     },
     "execution_count": 173,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# raplacing null values with empty string\n",
    "news_dataset.fillna('')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "id": "c900cb88",
   "metadata": {},
   "outputs": [],
   "source": [
    "# merging the author and news title\n",
    "news_dataset['content'] = news_dataset['author'] +' '+news_dataset['title']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "id": "7f825b07",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0        Darrell Lucus House Dem Aide: We Didn’t Even S...\n",
      "1        Daniel J. Flynn FLYNN: Hillary Clinton, Big Wo...\n",
      "2        Consortiumnews.com Why the Truth Might Get You...\n",
      "3        Jessica Purkiss 15 Civilians Killed In Single ...\n",
      "4        Howard Portnoy Iranian woman jailed for fictio...\n",
      "                               ...                        \n",
      "20795    Jerome Hudson Rapper T.I.: Trump a ’Poster Chi...\n",
      "20796    Benjamin Hoffman N.F.L. Playoffs: Schedule, Ma...\n",
      "20797    Michael J. de la Merced and Rachel Abrams Macy...\n",
      "20798    Alex Ansary NATO, Russia To Hold Parallel Exer...\n",
      "20799              David Swanson What Keeps the F-35 Alive\n",
      "Name: content, Length: 20800, dtype: object\n"
     ]
    }
   ],
   "source": [
    "print(news_dataset['content'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 176,
   "id": "90892c96",
   "metadata": {},
   "outputs": [],
   "source": [
    "# separthing data & labael\n",
    "X =news_dataset.drop(columns='label', axis=1)\n",
    "Y = news_dataset['label']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "id": "dfa3db44",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "          id                                              title  \\\n",
      "0          0  House Dem Aide: We Didn’t Even See Comey’s Let...   \n",
      "1          1  FLYNN: Hillary Clinton, Big Woman on Campus - ...   \n",
      "2          2                  Why the Truth Might Get You Fired   \n",
      "3          3  15 Civilians Killed In Single US Airstrike Hav...   \n",
      "4          4  Iranian woman jailed for fictional unpublished...   \n",
      "...      ...                                                ...   \n",
      "20795  20795  Rapper T.I.: Trump a ’Poster Child For White S...   \n",
      "20796  20796  N.F.L. Playoffs: Schedule, Matchups and Odds -...   \n",
      "20797  20797  Macy’s Is Said to Receive Takeover Approach by...   \n",
      "20798  20798  NATO, Russia To Hold Parallel Exercises In Bal...   \n",
      "20799  20799                          What Keeps the F-35 Alive   \n",
      "\n",
      "                                          author  \\\n",
      "0                                  Darrell Lucus   \n",
      "1                                Daniel J. Flynn   \n",
      "2                             Consortiumnews.com   \n",
      "3                                Jessica Purkiss   \n",
      "4                                 Howard Portnoy   \n",
      "...                                          ...   \n",
      "20795                              Jerome Hudson   \n",
      "20796                           Benjamin Hoffman   \n",
      "20797  Michael J. de la Merced and Rachel Abrams   \n",
      "20798                                Alex Ansary   \n",
      "20799                              David Swanson   \n",
      "\n",
      "                                                    text  \\\n",
      "0      House Dem Aide: We Didn’t Even See Comey’s Let...   \n",
      "1      Ever get the feeling your life circles the rou...   \n",
      "2      Why the Truth Might Get You Fired October 29, ...   \n",
      "3      Videos 15 Civilians Killed In Single US Airstr...   \n",
      "4      Print \\nAn Iranian woman has been sentenced to...   \n",
      "...                                                  ...   \n",
      "20795  Rapper T. I. unloaded on black celebrities who...   \n",
      "20796  When the Green Bay Packers lost to the Washing...   \n",
      "20797  The Macy’s of today grew from the union of sev...   \n",
      "20798  NATO, Russia To Hold Parallel Exercises In Bal...   \n",
      "20799    David Swanson is an author, activist, journa...   \n",
      "\n",
      "                                                 content  \n",
      "0      Darrell Lucus House Dem Aide: We Didn’t Even S...  \n",
      "1      Daniel J. Flynn FLYNN: Hillary Clinton, Big Wo...  \n",
      "2      Consortiumnews.com Why the Truth Might Get You...  \n",
      "3      Jessica Purkiss 15 Civilians Killed In Single ...  \n",
      "4      Howard Portnoy Iranian woman jailed for fictio...  \n",
      "...                                                  ...  \n",
      "20795  Jerome Hudson Rapper T.I.: Trump a ’Poster Chi...  \n",
      "20796  Benjamin Hoffman N.F.L. Playoffs: Schedule, Ma...  \n",
      "20797  Michael J. de la Merced and Rachel Abrams Macy...  \n",
      "20798  Alex Ansary NATO, Russia To Hold Parallel Exer...  \n",
      "20799            David Swanson What Keeps the F-35 Alive  \n",
      "\n",
      "[20800 rows x 5 columns]\n",
      "0        1\n",
      "1        0\n",
      "2        1\n",
      "3        1\n",
      "4        1\n",
      "        ..\n",
      "20795    0\n",
      "20796    0\n",
      "20797    0\n",
      "20798    1\n",
      "20799    1\n",
      "Name: label, Length: 20800, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(X)\n",
    "print(Y)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a682b39e",
   "metadata": {},
   "source": [
    "steming:\n",
    "\n",
    "steming process of reducing a word to its Root word\n",
    "\n",
    "example :\n",
    "\n",
    "actor , actress, acting --> act"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "id": "3d4ed0fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "port_stem = PorterStemmer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "id": "2f4e053a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def stemming(content):\n",
    "    stemmed_content = re.sub('[^a-zA-Z]',' ',content)\n",
    "    stemmed_content = stemmed_content.lower()\n",
    "    stemmed_content = stemmed_content.split()\n",
    "    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]\n",
    "    stemmed_content = ' '.join(stemmed_content)\n",
    "    return stemmed_content"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "id": "0bf2dce8",
   "metadata": {},
   "outputs": [],
   "source": [
    "news_dataset['content'] = news_dataset['content'].apply(str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "id": "9a0fe4a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0        Darrell Lucus House Dem Aide: We Didn’t Even S...\n",
      "1        Daniel J. Flynn FLYNN: Hillary Clinton, Big Wo...\n",
      "2        Consortiumnews.com Why the Truth Might Get You...\n",
      "3        Jessica Purkiss 15 Civilians Killed In Single ...\n",
      "4        Howard Portnoy Iranian woman jailed for fictio...\n",
      "                               ...                        \n",
      "20795    Jerome Hudson Rapper T.I.: Trump a ’Poster Chi...\n",
      "20796    Benjamin Hoffman N.F.L. Playoffs: Schedule, Ma...\n",
      "20797    Michael J. de la Merced and Rachel Abrams Macy...\n",
      "20798    Alex Ansary NATO, Russia To Hold Parallel Exer...\n",
      "20799              David Swanson What Keeps the F-35 Alive\n",
      "Name: content, Length: 20800, dtype: object\n"
     ]
    }
   ],
   "source": [
    "print(news_dataset['content'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "id": "7f45cfdf",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = news_dataset['content'].values\n",
    "Y = news_dataset['label'].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "id": "1226a158",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Darrell Lucus House Dem Aide: We Didn’t Even See Comey’s Letter Until Jason Chaffetz Tweeted It'\n",
      " 'Daniel J. Flynn FLYNN: Hillary Clinton, Big Woman on Campus - Breitbart'\n",
      " 'Consortiumnews.com Why the Truth Might Get You Fired' ...\n",
      " 'Michael J. de la Merced and Rachel Abrams Macy’s Is Said to Receive Takeover Approach by Hudson’s Bay - The New York Times'\n",
      " 'Alex Ansary NATO, Russia To Hold Parallel Exercises In Balkans'\n",
      " 'David Swanson What Keeps the F-35 Alive']\n"
     ]
    }
   ],
   "source": [
    "print(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "id": "09e25fc3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 0 1 ... 0 1 1]\n"
     ]
    }
   ],
   "source": [
    "print(Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 185,
   "id": "27957f06",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(20800,)"
      ]
     },
     "execution_count": 185,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 186,
   "id": "c5008060",
   "metadata": {},
   "outputs": [],
   "source": [
    "# converting the textual data to numerical data\n",
    "vectorizer = TfidfVectorizer()\n",
    "vectorizer.fit(X)\n",
    "\n",
    "X = vectorizer.transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "id": "7e7b3fdb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  (0, 21779)\t0.1815073237820376\n",
      "  (0, 21125)\t0.2702789788532812\n",
      "  (0, 20794)\t0.3444725998231082\n",
      "  (0, 17883)\t0.2246866205253603\n",
      "  (0, 12046)\t0.2990072566658748\n",
      "  (0, 11689)\t0.24971077104217873\n",
      "  (0, 10657)\t0.20488751270894362\n",
      "  (0, 10559)\t0.15453665306055495\n",
      "  (0, 9708)\t0.18427133691074812\n",
      "  (0, 7062)\t0.22914256323107624\n",
      "  (0, 5739)\t0.24818911212666975\n",
      "  (0, 5421)\t0.24818911212666975\n",
      "  (0, 5142)\t0.2990072566658748\n",
      "  (0, 4202)\t0.20576237430417024\n",
      "  (0, 3597)\t0.3101507937873876\n",
      "  (0, 843)\t0.2638990660507234\n",
      "  (1, 22152)\t0.30115379423155725\n",
      "  (1, 14123)\t0.1613179137611923\n",
      "  (1, 9471)\t0.19256474395011988\n",
      "  (1, 7862)\t0.7001643288235277\n",
      "  (1, 5105)\t0.26182005770192585\n",
      "  (1, 4004)\t0.19513664630807576\n",
      "  (1, 3261)\t0.37514222440000566\n",
      "  (1, 2864)\t0.15214163826848767\n",
      "  (1, 2362)\t0.292343693244767\n",
      "  :\t:\n",
      "  (20797, 10517)\t0.12650790266249798\n",
      "  (20797, 9743)\t0.20630992576636445\n",
      "  (20797, 5190)\t0.2090390876474378\n",
      "  (20797, 3159)\t0.14626936379698277\n",
      "  (20797, 2111)\t0.3170998027798729\n",
      "  (20797, 1372)\t0.2970786188328364\n",
      "  (20797, 1152)\t0.09998145136938631\n",
      "  (20797, 505)\t0.2861530832930508\n",
      "  (20798, 20324)\t0.11294792683766804\n",
      "  (20798, 17373)\t0.22239945672946546\n",
      "  (20798, 14573)\t0.42662749541285416\n",
      "  (20798, 13523)\t0.3088125110492585\n",
      "  (20798, 10055)\t0.12576444463220002\n",
      "  (20798, 9560)\t0.33667652496767764\n",
      "  (20798, 7155)\t0.42662749541285416\n",
      "  (20798, 1922)\t0.43947043690080384\n",
      "  (20798, 1233)\t0.29124605271938153\n",
      "  (20798, 942)\t0.26696017431840274\n",
      "  (20799, 21911)\t0.27218143719967247\n",
      "  (20799, 20127)\t0.1032319579412451\n",
      "  (20799, 19688)\t0.44315812085778317\n",
      "  (20799, 11026)\t0.45356383548562357\n",
      "  (20799, 5169)\t0.29843825192224444\n",
      "  (20799, 974)\t0.45077921399642495\n",
      "  (20799, 248)\t0.46998283498364773\n"
     ]
    }
   ],
   "source": [
    "print(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "id": "d6ca1266",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2,stratify=Y,random_state=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 189,
   "id": "4fd1fa96",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 190,
   "id": "afd0ddac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 190,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 191,
   "id": "0be44fc9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# accuracy score on the training data\n",
    "X_train_prediction = model.predict(X_train)\n",
    "training_data_accuracy = accuracy_score(X_train_prediction, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 192,
   "id": "c0d28133",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy score of the training data : 0.9906850961538461\n"
     ]
    }
   ],
   "source": [
    "print('Accuracy score of the training data :',training_data_accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 193,
   "id": "fcb9cf5a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# accuracy score on the test data\n",
    "X_test_prediction = model.predict(X_test)\n",
    "test_data_accuracy = accuracy_score(X_test_prediction, Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "id": "4816e908",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy score of the test data : 0.9848557692307692\n"
     ]
    }
   ],
   "source": [
    "print('Accuracy score of the test data :',test_data_accuracy)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9703f2dd",
   "metadata": {},
   "source": [
    "Making predictive system"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "id": "21a3695e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0]\n",
      "The news is Real\n"
     ]
    }
   ],
   "source": [
    "X_new = X_test[3]\n",
    "\n",
    "prediction = model.predict(X_new)\n",
    "print(prediction)\n",
    "\n",
    "if (prediction[0]==0):\n",
    "  print('The news is Real')\n",
    "else:\n",
    "  print('The news is Fake')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "id": "f3fbd6e6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n"
     ]
    }
   ],
   "source": [
    "print(Y_test[3])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
