import {defineStore} from 'pinia'

export const useFactoryState = defineStore('factory_state', {
    state: () => ({
        factoryList: [2333]
    }),
    actions: {
        install() {
            this.status = 'install';
        },
        uninstall() {
            this.status = 'uninstall';
        }
    }
})
