import mcid
import pandas as pd
import class_rule as cr


train = pd.read_csv('/Users/bowen/Desktop/lipid_plus_new/dataset/train_lipid_database.csv')
val = pd.read_csv('/Users/bowen/Desktop/lipid_plus_new/dataset/val_lipid_database.csv')
test = pd.read_csv('/Users/bowen/Desktop/lipid_plus_new/dataset/processed/test_nov.csv')
standard = pd.read_csv('/Users/bowen/Desktop/lipid_plus_new/dataset/standard_v2.csv')
test = cr.add_final_category(test)
standard = cr.add_final_category(standard)
test = cr.add_num_chain(test)
standard = cr.add_num_chain(standard)

train = mcid. ms2_norm(train)
val = mcid.ms2_norm(val)
test = mcid.ms2_norm(test)
standard = mcid.ms2_norm(standard)

# for df_name in ["train", "val", "test", "standard"]:
#     df = globals()[df_name]
#     globals()[df_name] = df[~((df['class'] == 'TG') & (df['num_peaks'] == 1))]


train.to_csv('/Users/bowen/Desktop/lipid_plus_new/dataset/processed/train_dec.csv')
val.to_csv('/Users/bowen/Desktop/lipid_plus_new/dataset/processed/val_dec.csv')
test.to_csv('/Users/bowen/Desktop/lipid_plus_new/dataset/processed/test_dec.csv')
standard.to_csv('/Users/bowen/Desktop/lipid_plus_new/dataset/processed/standard_dec.csv')


# train = mcid.process_ms2_df(train, decimal_point=0, neutral_loss=True, keep_intensity=False)
# val = mcid.process_ms2_df(val, decimal_point=0, neutral_loss=True, keep_intensity=True)
# test = mcid.process_ms2_df(test, decimal_point=0, neutral_loss=True, keep_intensity=True)
standard = mcid.process_ms2_df(standard, decimal_point=0, neutral_loss=True, keep_intensity=True)

# train.to_csv('/Users/bowen/Desktop/lipid_plus_new/dataset/processed/train_nl_noint.csv', index=False)
# val.to_csv('/Users/bowen/Desktop/lipid_plus_new/dataset/processed/val_nl_noint.csv', index=False)
# test.to_csv('/Users/bowen/Desktop/lipid_plus_new/dataset/processed/test_nl_noint.csv', index=False)
standard.to_csv('/Users/bowen/Desktop/lipid_plus_new/dataset/processed/standard_nl_noint.csv', index=False)
