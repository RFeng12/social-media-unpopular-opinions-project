# ["_all", "reddit.com", "news", "WTF", "funny", "Music", "worldnews", "Economics", "business", "AskReddit", "programming", 
# "linux", "technology", "hardware", "nsfw", "science", "politics", "atheism", "wikipedia", "pics", "philosophy", "gaming", 
# "sex", "feminisms", "obama", "space", "lolcats", "apple", "comics", "Libertarian", "gadgets", "self", "Marijuana", 
# "environment", "offbeat", "aww", "geek", "history", "web_design", "videos", "bestof", "Boobies", "happy", "sports", 
# "Health", "ruby", "egalitarian", "energy", "MensRights", "Anarchism", "movies", "worldpolitics", "psychology", 
# "entertainment", "photography", "canada", "religion", "PS3", "humor", "guns", "ronpaul", "PalinProblem", "Design",
# "Frugal", "Chinese", "economy", "food", "Art", "conspiracy", "women", "math", "BayuAditya", "Bacon", "firefox", "scifi", 
# "gossip", "bicycling", "netsec", "independent", "AmericanPolitics", "doctorwho", "australia", "video", "Pets", "lgbt",
# "beer", "writing", "ads", "auto", "socialism", "collapse", "CommonLaw", "Israel", "government", "iphone", "ukpolitics",
# "books", "joel", "opensource", "de", "WeAreTheMusicMakers", "software", "it", "cogsci", "USPE08", "usa", "Christianity",
# "Drugs", "celebrities", "Python", "WhiteMenGoneWild", "lists", "Military", "howto", "unitedkingdom", "Pictures", "compsci",
# "Buddhism", "MMA", "MapleLinks", "Freethought", "ja", "UFOs", "redditanonoymous", "scientology", "Sexy", "travel", "Green", 
# "xkcd", "AmericanGovernment", "japan", "Metal", 
# "anime", "Ubuntu", "DIY", "islam", "haskell", "idea", "Guitar", "hackers", "civilengineering", "astro", "ArtCrit"]
import read_subreddit_data
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import praw

numberposts = 30
dopairs = False
doClose = False
doRecs = True
doBest = False

fastBERT = True

comparison_tests = [
    ('news', 'gaming'),
    ('worldnews', 'news'),
    ('DIY', 'books'),
    ('Economics', 'business'),
    ('AmericanPolitics', 'business'),
    ('hackers', 'Python'),
    ('scientology', 'australia'),
    ('science', 'technology'),
    ('MMA', 'writing'),
    ('socialism', 'economy'),
    ('Economics', 'economy'),
    ('cogsci', 'Python'),
    ('Military', 'guns'),
    ('Drugs', 'Marijuana'),
    ('funny', 'humor'),
    ('Libertarian', 'environment'),
    ('space', 'photography'),
    ('space', 'technology'),
    ('Libertarian', 'Freethought'),
    ('religion', 'Buddhism'),
    ('movies', 'entertainment'),
    ('PS3', 'gaming'),
    ('AmericanPolitics', 'obama'),
    ('howto', 'DIY'),
    ('howto', 'obama'),
    ('socialism', 'Anarchism'),
    ('psychology', 'philosophy'),
    ('DIY', 'usa'),
    ('Chinese', 'usa'),
    ('Overwatch', 'usa'),
    ('Overwatch', 'Military'),
    ('Overwatch', 'AmericanGovernment'),
    ('Overwatch', 'gaming'),
    ('linux', 'gaming')
]
closest_tests = [
    'travel',
    'Art',
    'gaming',
    'business',
    'DIY',
    'UFOs',
    'Pets',
    'WeAreTheMusicMakers',
    'worldpolitics',
    'history',
    'feminisms',
    'Buddhism',
    'AskReddit',
    'NonCredibleDefense',
    'Genshin_Impact',
    'Overwatch'
]



if dopairs:
    for pair in comparison_tests:
        word1 = pair[0]
        word2 = pair[1]
        if word1 not in read_subreddit_data.SubsToProcess or word2 not in read_subreddit_data.SubsToProcess:
            print(word1, '-', word2, ' skipped')
            continue
        euclid_dist = read_subreddit_data.distance(word1, word2)
        cosine_dist = read_subreddit_data.cosine_dissimilarity(word1, word2)
        BERT_dist = read_subreddit_data.BERT_dissimilarity(word1, word2)

        print(word1, '-', word2, ' euclid: ', read_subreddit_data.get_rating_euclid(euclid_dist))
        print(word1, '-', word2, ' cosine: ', read_subreddit_data.get_rating_cosine(cosine_dist))
        if BERT_dist == -1:
            print(word1, '-', word2,' BERT SKIPPED')
        else:
            print(word1, '-', word2, ' BERT: ', read_subreddit_data.get_rating_BERT(BERT_dist))

    print('-------------------------------')



with open('connections_BERT') as file:
        connections_BERT = json.load(file)

with open('values_BERT') as file:
        distances_BERT_read = [0]
        for line in file:
            distances_BERT_read.append(float(line))
        distances_BERT_read = distances_BERT_read[1:]


if doClose:
    for test in closest_tests:
        if test not in read_subreddit_data.SubsToProcess:
            print(test, ' skipped')
            continue

        euclid_dist, euclid_name = read_subreddit_data.get_closest_to_sub_euclid(test)
        print(test, ' closest euclid dist: ', read_subreddit_data.get_rating_euclid(euclid_dist), ', ', euclid_name)

        cosine_dist, cosine_name = read_subreddit_data.get_closest_to_sub_cosine(test)
        print(test, ' closest cosine dist: ', read_subreddit_data.get_rating_cosine(cosine_dist), ', ', cosine_name)

        if fastBERT == True:
            bert_dist, bert_name = read_subreddit_data.get_closest_to_sub_BERT_fast(test, 20)
            print(test, ' close BERT dist: ', read_subreddit_data.get_rating_BERT(bert_dist), ', ', bert_name)
        
        else:
            for i in range(0, len(connections_BERT)):
                connection = connections_BERT[i]
                dist = distances_BERT_read[i]
                words = connection.split('-')
                if test == words[0]:
                    print(test, ' closest BERT dist: ', read_subreddit_data.get_rating_BERT(dist), ', ', words[1])
                    break
                if test == words[1]:
                    print(test, ' closest BERT dist: ', read_subreddit_data.get_rating_BERT(dist), ', ', words[0])
                    break
    

        print('top 5 user-based recs for ', test, ':')
        dict_, dict_unres = read_subreddit_data.get_user_dists(test)
        ind = 0
        for x in dict_.keys():
            print(x, 'at closeness', dict_[x])
            ind += 1
            if ind > 4:
                break

        print('top 5 user-based recs unrestricted for ', test, ':')
        ind = 0
        for x in dict_unres.keys():
            print(x, 'at closeness', dict_unres[x])
            ind += 1
            if ind > 4:
                break


if doRecs:
    for test in closest_tests:
        print('--------------------------------')
        if test not in read_subreddit_data.SubsToProcess:
            print(test, ' skipped')
            continue
        

        names, dists = read_subreddit_data.get_close_recs_euclid(test, 5)
        print('euclid recommendations for ', test, ':')
        out = ""
        for name in names:
            out += name + " "
        print(out)
        out = ""
        for dist in dists:
            out += str(read_subreddit_data.get_rating_euclid(dist)) + " "
        print(out)


        names, dists = read_subreddit_data.get_close_recs_cos(test, 5)
        print('cosine recommendations for ', test, ':')
        out = ""
        for name in names:
            out += name + " "
        print(out)
        out = ""
        for dist in dists:
            out += str(read_subreddit_data.get_rating_cosine(dist)) + " "
        print(out)


        if fastBERT == True:
            names, dists = read_subreddit_data.get_close_recs_BERT_fast(test, 5, 20)
            print('BERT recommendations for ', test, ':')
            out = ""
            for name in names:
                out += name + " "
            print(out)
            out = ""
            for dist in dists:
                out += str(read_subreddit_data.get_rating_BERT(dist)) + " "
            print(out)
        
        else:
            for i in range(0, len(connections_BERT)):
                connection = connections_BERT[i]
                dist = distances_BERT_read[i]
                words = connection.split('-')
                if test == words[0]:
                    print(test, ' closest BERT dist: ', read_subreddit_data.get_rating_BERT(dist), ', ', words[1])
                    break
                if test == words[1]:
                    print(test, ' closest BERT dist: ', read_subreddit_data.get_rating_BERT(dist), ', ', words[0])
                    break
    

        print('top 5 user-based recs for ', test, ':')
        dict_, dict_unres = read_subreddit_data.get_user_dists(test, True)
        ind = 0
        out1 = ""
        out2 = ""
        for x in dict_.keys():
            out1 += x + " "
            out2 += str(dict_[x]) + " "
            ind += 1
            if ind > 4:
                break
        print(out1)
       # print(out2)

        print('top 5 user-based recs unrestricted for ', test, ':')
        ind = 0
        out1 = ""
        out2 = ""
        for x in dict_unres.keys():
            out1 += x + " "
            out2 += str(dict_unres[x]) + " "
            ind += 1
            if ind > 4:
                break
        print(out1)
        #print(out2)

        print('top 5 user-based recs unormalized for ', test, ':')
        dict_, dict_unres = read_subreddit_data.get_user_dists(test, False)
        ind = 0
        out1 = ""
        out2 = ""
        for x in dict_.keys():
            out1 += x + " "
            out2 += str(dict_[x]) + " "
            ind += 1
            if ind > 4:
                break
        print(out1)
       # print(out2)

        print('top 5 user-based recs unrestricted unormalized for ', test, ':')
        ind = 0
        out1 = ""
        out2 = ""
        for x in dict_unres.keys():
            out1 += x + " "
            out2 += str(dict_unres[x]) + " "
            ind += 1
            if ind > 4:
                break
        print(out1)
       # print(out2)

        





if doBest:
    with open('connections_euclid') as json_file:
        connections_euclid = json.load(json_file)

    with open('connections_cosine') as file:
        connections_cosine = json.load(file)

    with open('connections_BERT') as file:
        connections_BERT = json.load(file)


    with open('values_cosine') as json_file:
        distances_cosine_read = [0]
        for line in json_file:
            distances_cosine_read.append(float(line))
        distances_cosine_read = distances_cosine_read[1:]

    with open('values_euclid') as file:
        distances_euclid_read = [0]
        for line in file:
            distances_euclid_read.append(float(line))
        distances_euclid_read = distances_euclid_read[1:]

    with open('values_BERT') as file:
        distances_BERT_read = [0]
        for line in file:
            distances_BERT_read.append(float(line))
        distances_BERT_read = distances_BERT_read[1:]






    for i in range(1, numberposts):
        print(connections_euclid[i], ' is at euclid distance ', read_subreddit_data.get_rating_euclid(distances_euclid_read[i]))

    for i in range(1, numberposts):
        print(connections_cosine[i], ' is at cosine distance ', read_subreddit_data.get_rating_cosine(distances_cosine_read[i]))

    for i in range(1, numberposts):
        print(connections_BERT[i], ' is at BERT distance ', read_subreddit_data.get_rating_BERT(distances_BERT_read[i]))
