import pandas as pd
import numpy as np
import seaborn as sns
import math
from ast import literal_eval
import collections
import os, glob, pickle
import librosa
import pathlib
from pydub import AudioSegment
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import time
from functools import lru_cache
import json
from tqdm.notebook import tqdm

from scipy.fft import fft, fftfreq, rfft, rfftfreq

from . import demo_analysis_new

import platform
path_delim = '\\' if platform.system() == 'Windows' else '/'

amt_of_players = 10

match_len={
    1:4,
    2:27,
    3:37,
    4:31}


emotion_with_keys={
  1:'интерес',
  2:'радость',
  3:'удивление',
  4:'горе',
  5:'гнев',
  6:'отвращение',
  7:'презрение',
  8:'страх',
  9:'стыд',
 10:'вина'
}

SAMPLING_RATE = 22050
pcs_len_sec = 3


@lru_cache(None)
def get_match_info(n_match):
    if n_match == 1:
        parsed_demo = './demos/3248aa5e-b344-40f5-8f83-4988a3b7141b_de_vertigo_128.csv'
    if n_match == 2:
        parsed_demo ='./demos/83fdd578-cb07-4c86-abb1-304cb0328b78_de_overpass_128.csv'
    if n_match == 3:
        parsed_demo = './demos/3e9849b7-304d-4017-96bc-41e7f0ce6a4e_de_vertigo_128.csv'
    if n_match == 4:
        parsed_demo = './demos/b504a2f2-82b6-4385-b0f3-ef9b88949655_de_mirage_128.csv'
        
    df = pd.read_csv(parsed_demo)
    tickrate=128 #default
    rounds_list = demo_analysis_new.get_round_stat(df,tickrate)

    return rounds_list

def get_players(n_match):
  
    players=['incr0ss','Softcore','humllet','faceitkirjke','SL4VAMARL0W']
    if n_match in [1,2]:
        players+=['___Tox1c___','giena1337','TheDefenderr','HOoL1GAN_','DENJKEZOR666'] #- VTB
    elif n_match==3:
        players+=['zhenn--','riddle','savagekx','Ka1n___','_SEGA'] #- GBCB
    elif n_match==4:
        players+=['zhenn--','riddle','savagekx','Ka1n___','RubinskiyRV'] #- GBCB
    
    return players

@lru_cache(None)
def get_game_context():
    game_context={}
    for n_match, _ in match_len.items():
        rounds_list = get_match_info(n_match)
        rounds={}
        players=get_players(n_match)
        for n_round,round_ in enumerate(rounds_list):
            round_data = round_[0].copy()
            round_data['users_self'] = round_data.parameters.apply(literal_eval).apply(lambda x: x.get('userid'))
            round_data['users_self'] = round_data['users_self'].apply(lambda x: players.index(x.split()[0]) if x and x.split()[0] in players else None)
            round_data['users_attacker'] = round_data.parameters.apply(literal_eval).apply(lambda x: x.get('attacker'))
            round_data['users_attacker'] = round_data['users_attacker'].apply(lambda x: players.index(x.split()[0]) if x and x.split()[0] in players else None)
            
            round_data['ms']=(((round_data['tick'] - round_data.tick.iloc[0])/128)*1000)
            rounds[n_round]=round_data
        game_context[n_match]=rounds
    return game_context


#разбиение всех аудио по 3 сек в соответствии с csv разметкой
def split_audio(path_to_audio,path_to_splitted_audio):
    for path in pathlib.Path(path_to_audio).iterdir():
        if path.is_dir():
            player_number = str(path)[str(path).rfind(path_delim)+1:].split('_')[0]
            for path_in in pathlib.Path(path).iterdir():
                str_path = str(path_in)
                name = str_path[str_path.rfind(path_delim)+1:]
                sound = AudioSegment.from_wav(path_in)
                i=0
                while i<=len(sound):
                    if i+3000>len(sound):
                        cut = sound[i:len(sound)+1]
                        if not os.path.exists(path_to_splitted_audio+path_delim+player_number+'_'+name[:-4]+ f'_{i}_{len(sound)}.wav'):
                            cut.export(path_to_splitted_audio+path_delim+player_number+'_'+name[:-4]+ f'_{i}_{len(sound)}.wav', format="wav")
                        break
                    cut = sound[i:i+3000]
                    if not os.path.exists(path_to_splitted_audio+path_delim+player_number+'_'+name[:-4]+ f'_{i}_{i+3000}.wav'):
                        cut.export(path_to_splitted_audio+path_delim+player_number+'_'+name[:-4]+ f'_{i}_{i+3000}.wav', format="wav")
                    i+=3000

@lru_cache(None)
def get_dict_with_emotions(file_path): #keys: match -> n_round -> num_player value: dataframe
    all_emt_dict={}
    col_emt_names=["start", "end", "emt_est_1", "str_emt_est_1", "emt_est_2", "str_emt_est_2", "emt_est_3", "str_emt_est_3"]
    for n_match, rounds_in_match in match_len.items():
        match_emt_cl={}
        for n_round in range(1, rounds_in_match + 1):
            players_emt={}
            res_emt={}
            for num_player in range(amt_of_players):
                if glob.glob(file_path + path_delim + f'{num_player}_match{n_match}_round{n_round}.csv'):
                    str_path=glob.glob(file_path + path_delim + f'{num_player}_match{n_match}_round{n_round}.csv')[0]
                    try:
                        df_s = pd.read_csv(str_path,names=col_emt_names)
                    except UnicodeDecodeError:
                        if platform.system() == 'Windows':
                            os.system("notepad " + str_path)
                        else:
                            os.system("nano " + str_path)                            
                        df_s = pd.read_csv(str_path,names=col_emt_names)
                    res_emt[num_player]=df_s

            match_emt_cl[n_round]=res_emt
        all_emt_dict[n_match]=match_emt_cl 
    return all_emt_dict

def find_majority(est_with_str):
    votes=[i[0] for i in est_with_str]
    strength=[i[1] for i in est_with_str]
    vote_count = Counter(votes)
    most_ = vote_count.most_common(1)
    if (most_[0][1]>= 2) and (most_[0][0] > 0):
        ids = [ind for ind in range(len(votes)) if votes[ind] == most_[0][0]]
        mean_str = round(np.array([strength[i] for i in ids]).mean(),1)
        return [most_[0][0], mean_str]
    else:
        if (most_[0][0] == 0) and (most_[0][1] == 2):
            return [max(votes), max(strength)]
        return [-1,-1]

def get_emotions(df): 
    
    player_emt = df.copy()
    for i in range(1,4):
        player_emt[f'est_{i}'] = list(zip(player_emt[f'emt_est_{i}'], player_emt[f'str_emt_est_{i}']))

    ss=[]
    for i in player_emt[['est_1','est_2','est_3']].values:
        ss.append(find_majority(i))
    player_emt['emt']=np.asarray(ss)[:,0]
    player_emt['emt'] = player_emt['emt'].apply(int)
    player_emt['str_emt']=np.asarray(ss)[:,1]
    
    emt_in_time={}
    for i in player_emt[['start','emt']].query('emt>0').emt.unique():
        emt_in_time[i]=[j[0] for j in player_emt[['start','emt']].query('emt>0').values if j[1]==i]
    return emt_in_time

def convert_dict(all_emt_dict):
    full_emt={} #key - emotion_type: value - [n_player,n_match,n_round,start_time]
    for key in emotion_with_keys.keys():
        full_emt[key]=[]
    
    for n_match, rounds_in_match in match_len.items():
        for n_round in range(1, rounds_in_match + 1):
            for n_player in range(amt_of_players):
                if all_emt_dict.get(n_match) is not None and \
                   all_emt_dict.get(n_match).get(n_round) is not None and \
                   all_emt_dict.get(n_match).get(n_round).get(n_player) is not None:
                        emt_in_time = get_emotions(all_emt_dict.get(n_match).get(n_round).get(n_player))
                        for e, start_time in emt_in_time.items():
                            full_emt[e].append([n_player,n_match,n_round,start_time])
    return full_emt

def split_annotations(full_emt,path_to_audio,test_size=0.2):
    train_annotations_list = []
    val_annotations_list = []
    
    X = []
    y = []
    info=[]
    
    for key_emt, values in full_emt.items():
        for vals in values:
            n_player,n_match,n_round,list_start_time=vals
            for start_time in list_start_time:
                file = glob.glob(path_to_audio + path_delim + f'{n_player}_match{n_match}_round{n_round}_{start_time}*.wav')
                X.append(file[0])
                y.append(key_emt)
                info.append([n_player,n_match,n_round,start_time])
                
    #не учитываем "стыд"       
    ind_to_del = y.index(9)
    y.pop(ind_to_del)
    X.pop(ind_to_del)
    info.pop(ind_to_del)
    
    X_train, X_val, y_train, y_val, info_train, info_val = train_test_split(X, y, info, test_size=test_size, random_state=42, shuffle=True, stratify=y)
    train_annotations_list = list(zip(X_train, y_train,info_train))
    val_annotations_list = list(zip(X_val, y_val, info_val))
    
    return train_annotations_list, val_annotations_list



def display_emt(full_dict_with_emt):
    for key, value in full_dict_with_emt.items():
      print(key,'-',np.array([len(i[3]) for i in value]).sum(),'   ',emotion_with_keys[key])


def create_onehot_tensor(label):
    y_onehot = torch.zeros(len(emotion_with_keys))
    y_onehot[label-1]=1
    return y_onehot


@lru_cache(None)
def get_context_vector(n_player,n_match,n_round,start_time):
    game_context = get_game_context()
    context_vector = np.zeros(12)
    bomb_interaction_events=['bomb_pickup','bomb_beginplant','bomb_planted','bomb_exploded','bomb_begindefuse','bomb_defused','bomb_abortplant']
    if n_round>=2:
        end_time = game_context[n_match][n_round-2].iloc[-1].ms
        if start_time==0 and n_round-3>=0:
            prev_round=game_context[n_match][n_round-3]
            start_ = prev_round.iloc[-1].ms-5000
            df = prev_round.query('ms>=@start_ and ms<=@start_+3000')
        elif start_time==3000 and n_round-3>=0:
            prev_round=game_context[n_match][n_round-3]
            start_ = game_context[n_match][n_round-3].iloc[-1].ms-2000
            df_1 = prev_round.query('ms>=@start_')
            df_2 = game_context[n_match][n_round-2].query('ms<=1000')
            df = df_1.append(df_2, ignore_index=True)
        elif start_time==3000 and n_round-3<0:
            df = game_context[n_match][n_round-2].query('ms<=1000')
        elif start_time-5000 >= end_time:
            df = game_context[n_match][n_round-2].query('ms>=@end_time-6000')   
        else:
            df = game_context[n_match][n_round-2].query('ms>=@start_time-5000 and ms<=@start_time-2000')
            
        #check events
        #self player
        if df.query('event=="player_hurt" and  users_self == @n_player').values.size:
            context_vector[0]=1
        if df.query('event=="player_death" and  users_self == @n_player').values.size:
            context_vector[1]=1
        if df.query('event=="player_blind" and  users_self == @n_player').values.size:
            context_vector[2]=1
        if df.query('event in @bomb_interaction_events and  users_self == @n_player').values.size:
            context_vector[3]=1
            
        #teammate player    
        if df.query('event=="player_hurt" and  users_self != @n_player and ((users_self < 5 and @n_player < 5) or (users_self >= 5 and @n_player >= 5))').values.size:
            context_vector[4]=1
        if df.query('event=="player_death" and  users_self != @n_player and ((users_self < 5 and @n_player < 5) or (users_self >= 5 and @n_player >= 5))').values.size:
            context_vector[5]=1
        if df.query('event in @bomb_interaction_events and  users_self != @n_player and ((users_self < 5 and @n_player < 5) or (users_self >= 5 and @n_player >= 5))').values.size:
            context_vector[6]=1
            
        #enemy player killed/hurted by player
        if df.query('event=="player_hurt" and  users_self != @n_player and users_attacker == @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[7]=1
        if df.query('event=="player_death" and  users_self != @n_player and users_attacker == @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[8]=1
            
        #enemy player killed/hurted by another player
        if df.query('event=="player_hurt" and  users_self != @n_player and users_attacker != @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[9]=1
        if df.query('event=="player_death" and  users_self != @n_player and users_attacker != @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[10]=1
        
        #enemy player interacted with bomb
        if df.query('event in @bomb_interaction_events and  users_self != @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[11]=1
            
    return context_vector



class BaseAudioSignalDataset(Dataset):

    def __init__(self, data_list, use_game_context, sampling_rate=SAMPLING_RATE, normalise=False, augment=False, augment_type='noise', **ignored_kwargs):
        self.data_list = data_list
        self.use_game_context = bool(use_game_context)
        self.sampling_rate = sampling_rate
        self.normalise = normalise
        self.augment = augment
        self.augment_type = augment_type

        if self.use_game_context:
            print("Preparing game context vectors")
            start = time.time()
            for _, _, info in tqdm(self.data_list):
                get_context_vector(*info)
            print(f"Finished in {(time.time() - start)/60:.2f} minutes")

    def __len__(self):
        return len(self.data_list)
  
    def load_wav(self, path):
        sound_1d_array, sr = librosa.load(path, sr=self.sampling_rate) # load audio to 1d array
        if self.normalise: #mean normalisation
            sound_1d_array = sound_1d_array - np.mean(sound_1d_array)

        if self.augment:
            sound_1d_array = self.augment_data(sound_1d_array, mode=self.augment_type, sr=sr)


        if sound_1d_array.shape[-1] < sr*pcs_len_sec:
            offset = sr*pcs_len_sec - sound_1d_array.shape[-1] 
            sound_1d_array = np.pad(sound_1d_array, (0, offset)) 
        return sound_1d_array

    def augment_data(self, sample, mode='noise', sr=None):
        if mode =='noise':
            y_noise = sample.copy()
            noise_amp = 0.005*np.random.uniform()*np.amax(y_noise)
            augmented = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])
        elif mode == 'pitch':
            y_pitch = sample.copy()
            bins_per_octave = 12
            pitch_pm = 2
            pitch_change = pitch_pm * 2*(np.random.uniform())
            augmented = librosa.effects.pitch_shift(y_pitch.astype('float64'),
                                      sr, n_steps=pitch_change,
                                      bins_per_octave=bins_per_octave)
        elif mode == None:
            augmented = sample
        else:
            raise NotImplemented(f"Augmentation type {mode} isn't implemented")

        return augmented

    def extract_features(self, path):
        x = self.load_wav(path)
        return torch.tensor(x)

    def __getitem__(self, index):
        item=self.data_list[index]
        x = self.extract_features(item[0])
        y = item[1] - 1
        
        if self.use_game_context:
            ctx = torch.from_numpy(get_context_vector(*item[2])).float()
        else:
            ctx = torch.tensor([])
        return x, ctx, y


class BaseSpectrogramDataset(BaseAudioSignalDataset):

    def __init__(self, 
                 data_list, 
                 use_game_context=True, 
                 sampling_rate=SAMPLING_RATE,
                 window_size=512, 
                 **ignored_kwargs):
        self.window_size = window_size
        self.hop_len = self.window_size // 2

        self.window_weights = np.hanning(self.window_size)[:, None]
        super().__init__(data_list, use_game_context, sampling_rate)

    @staticmethod
    def __visualize__(spec): 
        ax = sns.heatmap(spec)
        ax.invert_yaxis()

    def extract_features(self, path):
        _track = self.load_wav(path)
        spec = self.calculate_all_windows(_track)
        img = self.mapping_to_img(spec)
        return torch.tensor(img)

    @staticmethod
    def mapping_to_img(spectrum):
        maxval = np.min(spectrum) if np.abs(np.min(spectrum)) > np.abs(np.max(spectrum)) else np.max(spectrum)
        return 255*spectrum/maxval

    def __getitem__(self, index):
        item=self.data_list[index]
        x = self.extract_features(item[0])
        y = item[1] - 1
        
        if self.use_game_context:
            ctx = torch.from_numpy(get_context_vector(*item[2])).float()
        else:
            ctx = torch.tensor([])
        return x, ctx, y

    """
    For a typical speech recognition task, 
    a window of 20 to 30ms long is recommended.
    The overlap can vary from 25% to 75%.
    it is kept 50% for speech recognition.
    """
    def calculate_all_windows(self, audio):
        
        truncate_size = (len(audio) - self.window_size) % self.hop_len
        audio = audio[:len(audio) - truncate_size]

        nshape = (self.window_size, (len(audio) - self.window_size) // self.hop_len + 1)
        nhops = (audio.strides[0], audio.strides[0] * self.hop_len)
        
        windows = np.lib.stride_tricks.as_strided(audio, 
                                                  shape=nshape, 
                                                  strides=nhops)
        
        assert np.all(windows[:, 1] == audio[self.hop_len:(self.hop_len + self.window_size)])

        yf = np.fft.rfft(windows * self.window_weights, axis=0)
        yf = np.abs(yf)**2

        scaling_factor = np.sum(self.window_weights**2) * self.sampling_rate
        yf[1:-1, :] *= (2. / scaling_factor) 
        yf[(0,-1), :] /= scaling_factor

        xf = float(self.sampling_rate) / self.window_size * np.arange(yf.shape[0])

        indices = np.where(xf <= self.sampling_rate // 2)[0][-1] + 1
        return np.log(yf[:indices, :] + 1e-16)

def prepare_data(file_path,path_to_audio,path_to_splitted_audio,test_size):
    if not os.path.exists(path_to_splitted_audio):
        print('Start splitting audio to ',path_to_splitted_audio)
        os.makedirs(path_to_splitted_audio)
        split_audio(path_to_audio,path_to_splitted_audio)
        print('Finish splitting\n')
    else:
        print(f"{path_to_splitted_audio} exists; skip splitting")
    try:
        with open("prepared.json", "r") as file:
            train_list, val_list = json.load(file)
    except:
        full_dict_with_emt = convert_dict(get_dict_with_emotions(file_path))
        print('Emotion statistic:')
        display_emt(full_dict_with_emt)
        print('\nPrepare train and val lists')
        start = time.time()
        train_list, val_list = split_annotations(full_dict_with_emt,path_to_splitted_audio,test_size)
        print("It took: ", round((time.time()-start)/60,2)," minutes")
        print ('train size: ', len(train_list))
        print ('val size: ', len(val_list))
        try:
            with open("prepared.json", "w") as file:
                json.dump([train_list, val_list], file)
        except: print("pff")
       
    return train_list, val_list



def get_dataloader(file_path, path_to_audio, path_to_splitted_audio,
                   test_size,
                   use_game_context=False,
                   batch_size=32,
                   DatasetClass=BaseAudioSignalDataset,
                   num_workers=1):
  
    train_list, val_list = prepare_data(file_path,path_to_audio,path_to_splitted_audio,test_size)
    print('\nPrepare train dataset')
    train_dataset = DatasetClass(train_list,use_game_context=use_game_context, num_workers=num_workers)
    print('Prepare val dataset')
    val_dataset = DatasetClass(val_list,use_game_context=use_game_context, num_workers=num_workers)

    train_dataloader=DataLoader(
                train_dataset, batch_size=batch_size,
                num_workers=num_workers, shuffle=True,pin_memory=True)

    val_dataloader=DataLoader(
                val_dataset, batch_size=batch_size,
                num_workers=num_workers, shuffle=False,pin_memory=True)

    return train_dataloader,val_dataloader

#for ML methods
def get_data(data_list,use_game_context):

    amt_of_samples = sampling_rate*pcs_len_sec
        
    X = []
    Y = []
    ctx_ = []
        
    for item in data_list:
            
            sound_1d_array,_ = librosa.load(item[0])
            if sound_1d_array.size < amt_of_samples:
                offset = amt_of_samples - sound_1d_array.size 
                sound_1d_array = np.pad(sound_1d_array, (0, offset))

            X.append(sound_1d_array)
            y_onehot = create_onehot_tensor(item[1]).numpy()
            Y.append(y_onehot)
            if use_game_context:
                ctx_.append(get_context_vector(*item[2]))
                
    if use_game_context:
        return np.array(X), np.array(ctx_), np.array(Y)
    
    return np.array(X), np.array(Y)

def get_train_test(file_path,path_to_audio,path_to_splitted_audio,test_size,use_game_context=False):
    
    train_list, val_list = prepare_data(file_path,path_to_audio,path_to_splitted_audio,test_size)
    if use_game_context:
        print('Prepate train dataset')
        x_train, ctx_train, y_train = get_data(train_list,use_game_context=use_game_context)
        print('Prepate test dataset')
        x_test,ctx_test, y_test = get_data(val_list,use_game_context=use_game_context)
        return x_train,ctx_train,y_train,x_test,ctx_test,y_test
    else:
        print('Prepate train dataset')
        x_train,y_train = get_data(train_list,use_game_context=use_game_context)
        print('Prepate test dataset')
        x_test,y_test = get_data(val_list,use_game_context=use_game_context)
        return x_train,y_train,x_test,y_test
