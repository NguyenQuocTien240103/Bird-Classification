'use client'

import { Spinner } from '@/components/ui/shadcn-io/spinner'

export default function Loading() {
  return (
    <div className="h-screen w-screen flex items-center justify-center">
      <Spinner />
    </div>
  )
}
