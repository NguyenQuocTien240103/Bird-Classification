'use client'

import { Navbar } from "@/components/dashboard/navbar";
interface ContentLayoutProps {
  title: string;
  children: React.ReactNode;
}

export function ContentLayout({ title, children }: ContentLayoutProps) {
  return (
    <>
      <Navbar title={title} />
      <div className="flex-1 flex flex-col pt-4 pb-4 px-4 sm:px-8">{children}</div>
    </>
  );
}