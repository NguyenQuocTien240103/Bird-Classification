import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

interface User {
  id: number
  email: string
  is_active: boolean
  created_at: string
}

interface AuthState {
  user: User | null
  // token: string | null
  // isAuthenticated: boolean
  isLoading: boolean
  isLogout: boolean
  login: (email: string, password: string) => Promise<boolean>
  register: (email: string, password: string, confirm_password: string) => Promise<boolean>
  logout: () => void
  clearAuth: () => void
  getUserCurrent: () => Promise<boolean>
}

const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export const useAuthStore = create<AuthState>()(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      isLogout: false,
      login: async (email: string, password: string): Promise<boolean> => {
        set({ isLoading: true })
        try {
          const response = await fetch(`${API_URL}/auth/login`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            credentials: 'include',
            body: JSON.stringify({ email, password }),
          })
          if (response.ok) {
            const data = await response.json()
            const token = data.access_token

            // Get user info
            const userResponse = await fetch(`${API_URL}/user/me`, {
              method: 'GET',
              credentials: 'include',
            })

            if (userResponse.ok) {
              const userData = await userResponse.json()
              set({
                user: userData,
                isLogout: false,
                isLoading: false,
              })
              return true
            }
          }
          
          set({ isLoading: false })
          return false
        } catch (error) {   
          console.error('Login error:', error)
          set({ isLoading: false })
          return false
        }
      }, 
      register: async (email: string, password: string, confirm_password: string): Promise<boolean> => {
        set({ isLoading: true })
        try {
          const response = await fetch(`${API_URL}/auth/register`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password,confirm_password }),
          })

          if (response.ok) {
            set({ isLoading: false })   
            return true
          }
          
          set({ isLoading: false })
          return false
        } catch (error) {
          console.error('Registration error:', error)
          set({ isLoading: false })
          return false
        }
      },

      logout: async () => {
        set({
          user: null,
          isLogout: true,
        })
        try{
          const response = await fetch(`${API_URL}/auth/logout`, {
            method: 'GET',
            credentials: 'include',
          })
          if (!response.ok) {
            console.error('Logout failed')
          }
        }
        catch(error){
          console.error('Logout error:', error)
        }
      },

      clearAuth: () => {
        set({
          user: null,
          isLoading: false,
        })
      },
      getUserCurrent: async (): Promise<boolean> => {
        try{
          const userResponse = await fetch(`${API_URL}/user/me`, {
            method: 'GET',
            credentials: 'include',
          })
          if (userResponse.ok) {
            const userData = await userResponse.json()
            set({
              user: userData,
              isLogout: false,
              isLoading: false,
            })
            return true
          }
          return false
        }
        catch(error){
          console.error('Get user error:', error)
          set({
            user: null,
            isLogout: true,
          })
          return false
        }
      },
    }
  ),
)