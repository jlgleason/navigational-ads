import os

import dotenv
from bingads.service_client import ServiceClient
from bingads.authorization import *

dotenv.load_dotenv()

def authenticate(authorization_data):
    
    customer_service=ServiceClient(
        service='CustomerManagementService', 
        version=13,
        authorization_data=authorization_data, 
        environment=os.environ.get("ENVIRONMENT"),
    )

    # You should authenticate for Bing Ads services with a Microsoft Account.
    authenticate_with_oauth(authorization_data)
        
    # Set to an empty user identifier to get the current authenticated Bing Ads user,
    # and then search for all accounts the user can access.
    user=customer_service.GetUser(
        UserId=None
    ).User

    accounts=search_accounts_by_user_id(customer_service, user.Id)
    
    # For this example we'll use the first account.
    authorization_data.account_id=accounts['AdvertiserAccount'][0].Id
    authorization_data.customer_id=accounts['AdvertiserAccount'][0].ParentCustomerId
 
def authenticate_with_oauth(authorization_data):

    authentication=OAuthDesktopMobileAuthCodeGrant(
        client_id=os.environ.get("CLIENT_ID"),
        env=os.environ.get("ENVIRONMENT")
    )

    # It is recommended that you specify a non guessable 'state' request parameter to help prevent
    # cross site request forgery (CSRF). 
    authentication.state=os.environ.get("CLIENT_STATE")

    # Assign this authentication instance to the authorization_data. 
    authorization_data.authentication=authentication   

    # Register the callback function to automatically save the refresh token anytime it is refreshed.
    # Uncomment this line if you want to store your refresh token. Be sure to save your refresh token securely.
    authorization_data.authentication.token_refreshed_callback=save_refresh_token

    refresh_token=get_refresh_token()
    authorization_data.authentication.request_oauth_tokens_by_refresh_token(refresh_token)

def get_refresh_token():
    ''' 
    Returns a refresh token if stored locally.
    '''
    file=None
    try:
        file=open(os.environ.get("REFRESH_TOKEN"))
        line=file.readline()
        file.close()
        return line if line else None
    except IOError:
        if file:
            file.close()
        return None

def save_refresh_token(oauth_tokens):
    ''' 
    Stores a refresh token locally. Be sure to save your refresh token securely.
    '''
    with open(os.environ.get("REFRESH_TOKEN"),"w+") as file:
        file.write(oauth_tokens.refresh_token)
        file.close()
    return None

def search_accounts_by_user_id(customer_service, user_id):
    ''' 
    Search for account details by UserId.
    
    :param user_id: The Bing Ads user identifier.
    :type user_id: long
    :return: List of accounts that the user can manage.
    :rtype: Dictionary of AdvertiserAccount
    '''

    predicates={
        'Predicate': [
            {
                'Field': 'UserId',
                'Operator': 'Equals',
                'Value': user_id,
            },
        ]
    }

    accounts=[]

    page_index = 0
    PAGE_SIZE=100
    found_last_page = False

    while (not found_last_page):
        paging=set_elements_to_none(customer_service.factory.create('ns5:Paging'))
        paging.Index=page_index
        paging.Size=PAGE_SIZE
        search_accounts_response = customer_service.SearchAccounts(
            PageInfo=paging,
            Predicates=predicates
        )
        
        if search_accounts_response is not None and hasattr(search_accounts_response, 'AdvertiserAccount'):
            accounts.extend(search_accounts_response['AdvertiserAccount'])
            found_last_page = PAGE_SIZE > len(search_accounts_response['AdvertiserAccount'])
            page_index += 1
        else:
            found_last_page=True
    
    return {
        'AdvertiserAccount': accounts
    }

def set_elements_to_none(suds_object):
    # Bing Ads Campaign Management service operations require that if you specify a non-primitive, 
    # it must be one of the values defined by the service i.e. it cannot be a nil element. 
    # Since SUDS requires non-primitives and Bing Ads won't accept nil elements in place of an enum value, 
    # you must either set the non-primitives or they must be set to None. Also in case new properties are added 
    # in a future service release, it is a good practice to set each element of the SUDS object to None as a baseline. 

    for (element) in suds_object:
        suds_object.__setitem__(element[0], None)
    return suds_object
