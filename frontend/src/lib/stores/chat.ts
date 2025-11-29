import { create } from 'zustand'

interface Message {
  id: number
  conversation_id: number
  content: string
  is_user: boolean
  created_at: string
}

interface Conversation {
  id: number
  title: string
  user_id: number
  created_at: string
  updated_at: string
  messages: Message[]
}

interface ChatState {
  conversations: Conversation[]
  currentConversation: Conversation | null
  tempMessages: Message[]
  isLoading: boolean
  loadConversations: (token: string) => Promise<void>
  createConversation: (title: string, token: string) => Promise<Conversation | null>
  selectConversation: (conversation: Conversation, token: string) => Promise<void>
  sendMessage: (content: string, token?: string) => Promise<void>
  clearChat: () => void
  addTempMessage: (content: string, isUser: boolean) => void
}

const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: [],
  currentConversation: null,
  tempMessages: [],
  isLoading: false,

  loadConversations: async (token: string) => {
    set({ isLoading: true })
    try {
      const response = await fetch(`${API_URL}/conversations`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      })

      if (response.ok) {
        const conversations = await response.json()
        set({ conversations, isLoading: false })
      } else {
        set({ isLoading: false })
      }
    } catch (error) {
      console.error('Error loading conversations:', error)
      set({ isLoading: false })
    }
  },

  createConversation: async (title: string): Promise<Conversation | null> => {
    try {
      const response = await fetch(`${API_URL}/conversations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // 'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ title }),
      })

      if (response.ok) {
        const newConversation = await response.json()
        set((state) => ({
          conversations: [newConversation, ...state.conversations],
          currentConversation: newConversation,
        }))
        return newConversation
      }
      return null
    } catch (error) {
      console.error('Error creating conversation:', error)
      return null
    }
  },

  selectConversation: async (conversation: Conversation, token: string) => {
    set({ isLoading: true, currentConversation: conversation })
    try {
      const response = await fetch(`${API_URL}/conversations/${conversation.id}/messages`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      })

      if (response.ok) {
        const messages = await response.json()
        set((state) => ({
          currentConversation: { ...conversation, messages },
          isLoading: false,
        }))
      } else {
        set({ isLoading: false })
      }
    } catch (error) {
      console.error('Error loading conversation messages:', error)
      set({ isLoading: false })
    }
  },

  sendMessage: async (content: string, token?: string) => {
    const { currentConversation, tempMessages } = get()
    
    // Add user message immediately
    const userMessage: Message = {
      id: Date.now(),
      conversation_id: currentConversation?.id || 0,
      content,
      is_user: true,
      created_at: new Date().toISOString(),
    }

    if (token && currentConversation) {
      // Authenticated user - save to backend
      try {
        const response = await fetch(`${API_URL}/conversations/${currentConversation.id}/messages`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
          },
          body: JSON.stringify({
            content,
            is_user: true,
          }),
        })

        if (response.ok) {
          const savedMessage = await response.json()
          set((state) => ({
            currentConversation: state.currentConversation ? {
              ...state.currentConversation,
              messages: [...state.currentConversation.messages, savedMessage],
            } : null,
          }))

          // Add AI response placeholder (for demo purposes)
          const aiMessage: Message = {
            id: Date.now() + 1,
            conversation_id: currentConversation.id,
            content: "I'm a demo AI assistant. Your message has been saved to the database.",
            is_user: false,
            created_at: new Date().toISOString(),
          }

          const aiResponse = await fetch(`${API_URL}/conversations/${currentConversation.id}/messages`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${token}`,
            },
            body: JSON.stringify({
              content: aiMessage.content,
              is_user: false,
            }),
          })

          if (aiResponse.ok) {
            const savedAiMessage = await aiResponse.json()
            set((state) => ({
              currentConversation: state.currentConversation ? {
                ...state.currentConversation,
                messages: [...state.currentConversation.messages, savedAiMessage],
              } : null,
            }))
          }
        }
      } catch (error) {
        console.error('Error sending message:', error)
      }
    } else {
      // Guest user - store temporarily
      const aiMessage: Message = {
        id: Date.now() + 1,
        conversation_id: 0,
        content: "I'm a demo AI assistant. Please log in to save your conversation history.",
        is_user: false,
        created_at: new Date().toISOString(),
      }

      set({
        tempMessages: [...tempMessages, userMessage, aiMessage],
      })
    }
  },

  addTempMessage: (content: string, isUser: boolean) => {
    const message: Message = {
      id: Date.now(),
      conversation_id: 0,
      content,
      is_user: isUser,
      created_at: new Date().toISOString(),
    }

    set((state) => ({
      tempMessages: [...state.tempMessages, message],
    }))
  },

  clearChat: () => {
    set({
      conversations: [],
      currentConversation: null,
      tempMessages: [],
    })
  },
}))