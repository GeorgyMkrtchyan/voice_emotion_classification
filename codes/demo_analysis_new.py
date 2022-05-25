import ast
import numpy as np

MIN_ROUND_DURATION = 10
NUMBER_OF_PLAYERS = 10


def get_monotonic_tail(v):
    tail = v[-1:]
    elem_pred = tail[-1] if len(tail) > 0 else None
    for elem in v[-2::-1]:
        if elem < elem_pred:
            tail = [elem] + tail
            elem_pred = elem
        else:
            break
    return tail


def split_match_into_rounds(df, tickrate):
    df.loc[df.shape[0]]=[df.iloc[-1].tick+1,'round_officially_ended','{}'] 

    start_i = 0

    rounds = []
    key_events = ['round_prestart', 'round_start', 'round_freeze_end', 'round_end']
    key_events_orders = {e: i for i, e in enumerate(key_events)}

    for i, off_end_i in enumerate(df[df['event'] == 'round_officially_ended'].index):
        df_tmp = df.loc[start_i:off_end_i, :]
	
        if (i==(len(df[df['event']=='round_officially_ended'].index)-1)):
            ind_to_skip = df_tmp[df_tmp.event=='round_freeze_end'].index[1]
            df_tmp = df_tmp.query("index != @ind_to_skip")

        key_idx = df_tmp[df_tmp['event'].isin(key_events)].index
        round_tail_orders = get_monotonic_tail([key_events_orders[e] for e in df_tmp.loc[key_idx, 'event']])

        round_info = {'round_control_points': df_tmp.loc[key_idx, 'event'].values}

        if 'weapon_fire' in df_tmp['event'].values and 'player_footstep' in df_tmp['event'].values:
            if key_events_orders['round_freeze_end'] in round_tail_orders:
                tail_len = len(round_tail_orders)
                round_df = df_tmp.loc[key_idx[-tail_len]:]

                effective_time = (round_df['tick'].iloc[-1] - round_df[round_df['event'] == 'round_freeze_end']['tick'].iloc[0]) / tickrate
                effective_time_check = effective_time > MIN_ROUND_DURATION
                round_info['effective_time'] = effective_time

                time_limit_check = True
                if key_events_orders['round_start'] in round_tail_orders:
                    item = ast.literal_eval(round_df[round_df['event'] == 'round_start']['parameters'].iloc[0])
                    time_limit_check = item['timelimit'] == '115'

                footstep_users = get_footstep_players(round_df)
                number_of_users_check = len(footstep_users) == NUMBER_OF_PLAYERS
                round_info['round_footstep_players'] = sorted(footstep_users)

                if effective_time_check and time_limit_check and number_of_users_check:
                    rounds.append((round_info, round_df))
            else:
                footstep_users = get_footstep_players(df_tmp)
                pos_T, pos_CT = get_round_initial_positions(df_tmp, footstep_users)
                if is_players_initial(pos_T, pos_CT):
                    footsteps = df_tmp[df_tmp['event'] == 'player_footstep']
                    round_df = df_tmp.loc[footsteps.index[0]:]

                    effective_time = (round_df['tick'].iloc[-1] - round_df['tick'].iloc[0]) / tickrate
                    effective_time_check = effective_time > MIN_ROUND_DURATION
                    round_info['effective_time'] = effective_time

                    number_of_users_check = len(footstep_users) == NUMBER_OF_PLAYERS
                    round_info['round_footstep_players'] = sorted(footstep_users)

                    if effective_time_check and number_of_users_check:
                        rounds.append((round_info, round_df))

        start_i = off_end_i
    return rounds

# WEAPONS
KNIFES = [
    'knife_t',
    'knife_egg',
    'knife_ghost',
    'knife_bayonet',
    'knife_butterfly',
    'knife_falchion',
    'knife_flip',
    'knife_gut',
    'knife_tactical',
    'knife_karambit',
    'knife_m9_bayonet',
    'knife_push',
    'knife_survival_bowie',
    'knife_ursus',
    'knife_gypsy_jackknife',
    'knife_stiletto',
    'knife_widowmaker']
KNIFES += [s[6:] for s in KNIFES]
KNIFES += ['knife', 'knifegg']
KNIFES += ['weapon_'+s for s in KNIFES]

PISTOLS = ['hkp2000', 'usp_silencer', 'glock', 'p250', 'fiveseven', 'tec9', 'cz75a', 'elite', 'deagle', 'revolver']
PISTOLS += ['weapon_'+s for s in PISTOLS]

SMGS = ['mp9', 'mac10', 'bizon', 'mp7', 'ump45', 'p90', 'mp5sd']
SMGS += ['weapon_'+s for s in SMGS]

RIFLES = ['famas', 'galilar', 'm4a1', 'm4a4', 'm4a1_silencer', 'ak47', 'aug', 'sg556', 'ssg08', 'awp', 'scar20', 'g3sg1']
RIFLES += ['weapon_'+s for s in RIFLES]

HEAVY = ['nova', 'm249', 'xm1014', 'mag7', 'sawedoff', 'negev']
HEAVY += ['weapon_'+s for s in HEAVY]

GRENADES = ['hegrenade', 'incgrenade', 'smokegrenade', 'flashbang', 'decoy', 'molotov']
GRENADES += ['weapon_'+s for s in GRENADES]

GEAR = ['taser']
GEAR += ['weapon_'+s for s in GEAR]


def get_round_weapons_fired(df):
    weapons = set()
    for p in df[df['event'] == 'weapon_fire']['parameters']:
        weapons.add(ast.literal_eval(p)['weapon'])
    return weapons


def is_round_pistol(df, weapons=None):
    if weapons is None:
        weapons = get_round_weapons_fired(df)
    return len(weapons.intersection(set(SMGS + RIFLES + HEAVY))) == 0 and \
        len(weapons.intersection(set(PISTOLS))) > 0


def get_match_players(df):
    users = set()
    for e in df[df['event'] == 'player info']['parameters']:
        item = ast.literal_eval(e)
        if item['fakeplayer'] == '0':
            users.add(item['name'])
    return users


def get_footstep_players(df):
    users = set()
    for e in df[df['event'] == 'player_footstep']['parameters']:
        item = ast.literal_eval(e)
        user_name = item['userid'].split(' (id:')[0]
        users.add(user_name)
    return users


def get_round_initial_positions(df, round_users):
    positions_T = []
    positions_CT = []
    users = round_users.copy()
    for e in df[df['event'] == 'player_footstep']['parameters']:
        item = ast.literal_eval(e)
        user_found = None
        for u in users:
            if u == item['userid'][:len(u)]:
                if 'userid position' in item:
                    pos_coordinates = ast.literal_eval(item['userid position'])
                    if 'userid team' in item:
                        if item['userid team'] == 'T':
                            positions_T.append(pos_coordinates)
                            user_found = u
                            break
                        if item['userid team'] == 'CT':
                            positions_CT.append(pos_coordinates)
                            user_found = u
                            break
        if user_found is not None:
            users.remove(user_found)
        if users == []:
            break
    return positions_T, positions_CT


def is_position_initial(pos, team_type):
    x, y, z = pos
    if team_type == 'T':
        # if not (x < 1400 and 100 < y < 900):
        #     print(team_type, x, y)
        return x < -1400 and 100 < y < 900
    elif team_type == 'CT':
        # if not (2200 < x and 1600 < y < 2500):
        #     print(team_type, x, y)
        return 2200 < x and 1600 < y < 2500
    else:
        return None


def is_players_initial(positions_T, positions_CT):
    if len(positions_T) == 0 or len(positions_CT) == 0:
        return False

    p = [is_position_initial(pos, 'T') for pos in positions_T]
    p += [is_position_initial(pos, 'CT') for pos in positions_CT]

    return np.all(p)


def get_round_stat(df, tickrate):
    rounds = []

    for round_info, round_df in split_match_into_rounds(df, tickrate):
        if 'round_freeze_end' in round_df['event'].values:
            local_begin = round_df[round_df['event'] == 'round_freeze_end'].index[0]
        else:
            local_begin = round_df.index[0]

        if 'round_end' in round_df['event'].values:
            local_end = round_df[round_df['event'] == 'round_end'].index[0]
        else:
            local_end = round_df.index[-1]

        round_local = round_df.loc[local_begin:local_end]
        weapons = get_round_weapons_fired(round_local)
        round_info['weapons'] = weapons
        round_info['is_pistol'] = is_round_pistol(round_local, weapons)

        rounds.append((round_df, round_info))

    return rounds
