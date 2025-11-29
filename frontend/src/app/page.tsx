'use client'

import { ProfileForm } from '@/components/dashboard/profile-form';
import { ContentLayout } from "@/components/dashboard/content-layout";

export default function AccountPage() {

  return (
    <ContentLayout title="Phân loại chim">
      <div className="mt-6">
        <ProfileForm />
      </div>
    </ContentLayout>
  )
}