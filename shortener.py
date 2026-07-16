import re

def extract_package_id(input_text):
    if input_text.startswith("https://play.google.com/store/apps/details?"):
        match = re.search(r'\bid=([^\s&#]+)', input_text)
        if match:
            return match.group(1)
        else:
            return None
    else:
        return input_text.strip()
    
# input1 = "https://play.google.com/store/apps/details?id=com.tokopedia.tkpd&hl=id&gl=US"
# input1 = "https://play.google.com/store/apps/details?id=com.supercell.clashofclans&hl=id&gl=US"
# input1="https://play.google.com/store/apps/details?id=com.scopely.monopolygo&pcampaignid=web_share"
# input1="https://play.google.com/store/apps/details?id=com.YoStarEN.Arknights&pcampaignid=merch_published_cluster_promotion_battlestar_browse_all_games&hl=id&gl=US"
# input1 = "https://play.google.com/store/apps/details?id=com.mobile.legends&hl=id"
# input2 = "com.mobile.legends"

# package_id1 = extract_package_id(input1)
# package_id2 = extract_package_id(input2)

# print("Package ID dari input pertama:", package_id1)
# print("Package ID dari input kedua:", package_id2)
